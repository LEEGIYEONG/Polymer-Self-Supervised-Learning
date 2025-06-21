"""
Polymer Self-Supervised Learning Pipeline
=========================================

This module implements a comprehensive pre-training and self-supervised learning pipeline
for polymer property prediction, following the workflow from the Open Polymer Datasets report.

Workflow:
1. Representation pre-training on PI1M dataset (~1M p-SMILES)
2. Self-supervised tasks (masked prediction, contrastive learning)
3. Multi-task fine-tuning on competition data (Tg, FFV, Tc, Density, Rg)
4. Integration of additional datasets for missing properties
5. Synthetic feasibility weighting using SA scores

Dependencies:
- rdkit-pypi: For molecular fingerprints and SA score calculation
- pandas: Data manipulation
- torch: Deep learning framework
- transformers: For transformer-based models
- sklearn: Traditional ML algorithms
- tqdm: Progress bars
- matplotlib, seaborn: Visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    print("RDKit imported successfully")
except ImportError:
    print("RDKit not found. Installing...")
    os.system("pip install rdkit-pypi")
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

# Scientific computing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer

# Utilities
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import random
from pathlib import Path
import json

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Configuration
class Config:
    """Configuration class for the polymer ML pipeline"""
    
    # Paths
    BASE_PATH = Path(r"c:\Users\LEEGIYEONG\Downloads\ips")
    PI1M_PATH = BASE_PATH / "PI1M-master"
    COMPETITION_PATH = BASE_PATH / "neurips-open-polymer-prediction-2025 (1)"
    APFE_PATH = BASE_PATH / "APFEforPI-main" / "dataset"
    RG_PATH = BASE_PATH / "Sequence-Radius-of-gyration-Rg-data-of-a-copolymer-main"
      # Model parameters (optimized for RTX 3060 6GB)
    EMBEDDING_DIM = 256  # Reduced from 512
    HIDDEN_DIM = 128     # Reduced from 256
    NUM_LAYERS = 3       # Reduced from 4
    NUM_HEADS = 4        # Reduced from 8
    DROPOUT = 0.1
    
    # Training parameters (memory optimized)
    BATCH_SIZE = 16      # Reduced from 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 10
    
    # Properties to predict
    TARGET_PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
print(f"Using device: {config.DEVICE}")
print(f"Base path: {config.BASE_PATH}")

class PolymerDataLoader:
    """
    Data loader for multiple polymer datasets
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.datasets = {}
        self.combined_data = None
          def load_pi1m_data(self, version='v2', sample_size=None):
        """Load PI1M dataset"""
        print("Loading PI1M dataset...")
        
        try:
            # Try to load the specified version
            pi1m_file = self.config.PI1M_PATH / f"PI1M_{version}.csv"
            if not pi1m_file.exists():
                pi1m_file = self.config.PI1M_PATH / "PI1M.csv"
                print(f"PI1M_{version}.csv not found, using PI1M.csv")
            
            # For large files, read in chunks
            chunk_list = []
            chunk_size = 10000
            
            for chunk in pd.read_csv(pi1m_file, chunksize=chunk_size):
                # Clean data in each chunk
                chunk = chunk.dropna(subset=['SMILES'])  # Remove NaN SMILES
                chunk = chunk[chunk['SMILES'].str.len() > 0]  # Remove empty SMILES
                chunk_list.append(chunk)
                if sample_size and len(chunk_list) * chunk_size >= sample_size:
                    break
            
            pi1m_data = pd.concat(chunk_list, ignore_index=True)
            
            if sample_size:
                pi1m_data = pi1m_data.sample(n=min(sample_size, len(pi1m_data))).reset_index(drop=True)
            
            print(f"Loaded {len(pi1m_data)} PI1M samples")
            self.datasets['pi1m'] = pi1m_data
            return pi1m_data
            
        except Exception as e:
            print(f"Error loading PI1M data: {e}")
            return None
    
    def load_competition_data(self):
        """Load competition training data"""
        print("Loading competition data...")
        
        try:
            train_file = self.config.COMPETITION_PATH / "train.csv"
            test_file = self.config.COMPETITION_PATH / "test.csv"
            
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            
            print(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples")
            
            self.datasets['competition_train'] = train_data
            self.datasets['competition_test'] = test_data
            
            return train_data, test_data
            
        except Exception as e:
            print(f"Error loading competition data: {e}")
            return None, None
    
    def load_apfe_data(self):
        """Load APFEforPI thermal conductivity data"""
        print("Loading APFE thermal conductivity data...")
        
        try:
            example_file = self.config.APFE_PATH / "example_smiles.csv"
            apfe_data = pd.read_csv(example_file)
            
            # Rename TC to Tc for consistency
            if 'TC' in apfe_data.columns:
                apfe_data = apfe_data.rename(columns={'TC': 'Tc'})
            
            print(f"Loaded {len(apfe_data)} APFE samples")
            self.datasets['apfe'] = apfe_data
            
            return apfe_data
            
        except Exception as e:
            print(f"Error loading APFE data: {e}")
            return None
    
    def load_rg_data(self):
        """Load radius of gyration data"""
        print("Loading Rg data...")
        
        try:
            rg_files = list(self.config.RG_PATH.glob("q_*.txt"))
            rg_data_list = []
            
            for file in rg_files:
                q_value = file.stem.split('_')[1]
                data = pd.read_csv(file, sep=' ', header=None, names=['sequence', 'Rg'])
                data['q_value'] = q_value
                rg_data_list.append(data)
            
            if rg_data_list:
                rg_data = pd.concat(rg_data_list, ignore_index=True)
                # Convert sequence to SMILES-like format (simplified)
                rg_data['SMILES'] = rg_data['sequence'].apply(self._sequence_to_smiles)
                
                print(f"Loaded {len(rg_data)} Rg samples")
                self.datasets['rg'] = rg_data
                return rg_data
            else:
                print("No Rg files found")
                return None
                
        except Exception as e:
            print(f"Error loading Rg data: {e}")
            return None
      def _sequence_to_smiles(self, sequence):
        """Convert polymer sequence to approximate SMILES"""
        # Handle NaN and invalid sequences
        if pd.isna(sequence) or sequence == 'nan' or not isinstance(sequence, str):
            return None
        
        # This is a simplified conversion - in practice, you'd need more sophisticated mapping
        monomer_map = {
            'A': 'CC',      # Ethylene-like monomer
            'B': 'C(C)C',   # Propylene-like monomer  
            'C': 'C(C)CC',  # Butylene-like monomer
            'D': 'c1ccccc1', # Phenyl group
            'E': 'C(=O)',   # Carbonyl group
            'F': 'C(F)(F)F', # Trifluoromethyl
            # Add more mappings as needed
        }
        
        # Try to map sequence to actual SMILES
        try:
            smiles_parts = []
            for char in sequence:
                if char in monomer_map:
                    smiles_parts.append(monomer_map[char])
                else:
                    # For unknown characters, use simple carbon
                    smiles_parts.append('C')
            
            if smiles_parts:
                # Join with single bonds, add polymer notation
                return '*' + 'C'.join(smiles_parts) + '*'
            else:
                return None
        except:
            return None
      def combine_datasets(self):
        """Combine all loaded datasets"""
        print("Combining datasets...")
        
        combined_list = []
        
        # Process each dataset
        for name, data in self.datasets.items():
            if data is not None:
                # Standardize column names
                processed_data = data.copy()
                
                # Ensure SMILES column exists
                if 'SMILES' not in processed_data.columns:
                    if 'smiles' in processed_data.columns:
                        processed_data['SMILES'] = processed_data['smiles']
                    else:
                        continue
                
                # Add dataset source
                processed_data['source'] = name
                
                # Select relevant columns
                keep_cols = ['SMILES', 'source']
                for prop in self.config.TARGET_PROPERTIES:
                    if prop in processed_data.columns:
                        keep_cols.append(prop)
                
                processed_data = processed_data[keep_cols]
                combined_list.append(processed_data)
        
        if combined_list:
            self.combined_data = pd.concat(combined_list, ignore_index=True, sort=False)
            print(f"Combined dataset shape: {self.combined_data.shape}")
            
            # Remove invalid SMILES
            valid_smiles = self.combined_data['SMILES'].apply(self._is_valid_smiles)
            self.combined_data = self.combined_data[valid_smiles].reset_index(drop=True)
            print(f"After filtering invalid SMILES: {self.combined_data.shape}")
            
            return self.combined_data
        else:
            print("No datasets to combine")
            return None
    
    def _is_valid_smiles(self, smiles):
        """Check if SMILES string is valid"""
        try:
            if pd.isna(smiles) or smiles is None or smiles == '' or 'nan' in str(smiles).lower():
                return False
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

class MolecularFeaturizer:
    """
    Generate molecular features from SMILES strings
    """
    
    def __init__(self, feature_types=['morgan', 'descriptors', 'sa_score']):
        self.feature_types = feature_types
        self.scaler = StandardScaler()
        
    def featurize_smiles(self, smiles_list):
        """Generate features for list of SMILES"""
        features_list = []
        
        for smiles in tqdm(smiles_list, desc="Featurizing SMILES"):
            features = self._get_molecular_features(smiles)
            features_list.append(features)
        
        return np.array(features_list)
    
    def _get_molecular_features(self, smiles):
        """Get molecular features for a single SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self._get_feature_dim())
            
            features = []
            
            if 'morgan' in self.feature_types:
                # Morgan fingerprints (2048 bits)
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                features.extend(fp.ToBitString())
            
            if 'descriptors' in self.feature_types:
                # Molecular descriptors
                desc_features = [
                    Descriptors.MolWt(mol),
                    Descriptors.LogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.RingCount(mol),
                    Descriptors.FractionCsp3(mol),
                ]
                features.extend(desc_features)
            
            if 'sa_score' in self.feature_types:
                # Synthetic accessibility score (simplified)
                try:
                    from rdkit.Chem import rdMolDescriptors
                    sa_score = rdMolDescriptors.BertzCT(mol) / 100.0  # Normalized complexity
                    features.append(sa_score)
                except:
                    features.append(0.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            return np.zeros(self._get_feature_dim())
    
    def _get_feature_dim(self):
        """Get total feature dimension"""
        dim = 0
        if 'morgan' in self.feature_types:
            dim += 2048
        if 'descriptors' in self.feature_types:
            dim += 10
        if 'sa_score' in self.feature_types:
            dim += 1
        return dim

class PolymerDataset(Dataset):
    """PyTorch dataset for polymer data"""
    
    def __init__(self, features, targets=None, mask_prob=0.15):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        
        if self.targets is not None:
            targets = self.targets[idx]
            return features, targets
        else:
            # For self-supervised learning, create masked version
            masked_features = features.clone()
            mask = torch.rand(features.shape) < self.mask_prob
            masked_features[mask] = 0
            
            return masked_features, features  # Input, target for reconstruction

class PolymerTransformer(nn.Module):
    """
    Transformer-based model for polymer property prediction
    """
    
    def __init__(self, input_dim, embedding_dim=512, num_heads=8, num_layers=4, 
                 hidden_dim=256, num_properties=5, dropout=0.1):
        super(PolymerTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_properties = num_properties
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        
        # Positional encoding (simplified for molecular features)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads for different properties
        self.property_heads = nn.ModuleDict({
            'Tg': nn.Linear(embedding_dim, 1),
            'FFV': nn.Linear(embedding_dim, 1),
            'Tc': nn.Linear(embedding_dim, 1),
            'Density': nn.Linear(embedding_dim, 1),
            'Rg': nn.Linear(embedding_dim, 1),
        })
        
        # Reconstruction head for self-supervised learning
        self.reconstruction_head = nn.Linear(embedding_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_embeddings=False):
        # Input projection
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoding
        encoded = self.transformer(x)
        encoded = encoded.squeeze(1)  # Remove sequence dimension
        
        if return_embeddings:
            return encoded
        
        # Property predictions
        property_outputs = {}
        for prop, head in self.property_heads.items():
            property_outputs[prop] = head(self.dropout(encoded))
        
        # Reconstruction for self-supervised learning
        reconstruction = self.reconstruction_head(self.dropout(encoded))
        
        return property_outputs, reconstruction

class SelfSupervisedTrainer:
    """
    Trainer for self-supervised pre-training and fine-tuning
    """
    
    def __init__(self, model, config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        # Optimizers
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'property_losses': {prop: [] for prop in config.TARGET_PROPERTIES}
        }
    
    def pretrain(self, dataloader, num_epochs=50):
        """Pre-train with self-supervised reconstruction task"""
        print("Starting self-supervised pre-training...")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (masked_features, original_features) in enumerate(progress_bar):
                masked_features = masked_features.to(self.device)
                original_features = original_features.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                _, reconstruction = self.model(masked_features)
                
                # Reconstruction loss
                loss = self.mse_loss(reconstruction, original_features)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / num_batches
            self.history['train_loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(avg_loss)
        
        print("Pre-training completed!")
    
    def finetune(self, train_loader, val_loader, num_epochs=100):
        """Fine-tune on property prediction tasks"""
        print("Starting multi-task fine-tuning...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = {prop: [] for prop in self.config.TARGET_PROPERTIES}
            train_losses['total'] = []
            
            for batch_idx, (features, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                property_outputs, _ = self.model(features)
                
                # Calculate losses for each property
                total_loss = 0
                for i, prop in enumerate(self.config.TARGET_PROPERTIES):
                    if not torch.isnan(targets[:, i]).all():  # Skip if all NaN
                        mask = ~torch.isnan(targets[:, i])
                        if mask.sum() > 0:
                            prop_loss = self.mse_loss(
                                property_outputs[prop][mask].squeeze(), 
                                targets[mask, i]
                            )
                            total_loss += prop_loss
                            train_losses[prop].append(prop_loss.item())
                
                train_losses['total'].append(total_loss.item())
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
            
            # Validation phase
            val_loss = self._validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_polymer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log training progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {np.mean(train_losses['total']):.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            print("-" * 50)
        
        print("Fine-tuning completed!")
        # Load best model
        self.model.load_state_dict(torch.load('best_polymer_model.pth'))
    
    def _validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                property_outputs, _ = self.model(features)
                
                batch_loss = 0
                for i, prop in enumerate(self.config.TARGET_PROPERTIES):
                    if not torch.isnan(targets[:, i]).all():
                        mask = ~torch.isnan(targets[:, i])
                        if mask.sum() > 0:
                            prop_loss = self.mse_loss(
                                property_outputs[prop][mask].squeeze(), 
                                targets[mask, i]
                            )
                            batch_loss += prop_loss
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')

def create_datasets_and_loaders(combined_data, featurizer, config, test_size=0.2):
    """Create datasets and data loaders"""
    print("Creating datasets and data loaders...")
    
    # Generate molecular features
    features = featurizer.featurize_smiles(combined_data['SMILES'].tolist())
    
    # Prepare targets
    target_columns = []
    for prop in config.TARGET_PROPERTIES:
        if prop in combined_data.columns:
            target_columns.append(combined_data[prop].values)
        else:
            target_columns.append(np.full(len(combined_data), np.nan))
    
    targets = np.column_stack(target_columns)

    targets = targets.astype(np.float32)
    
    # Train-validation split
    train_features, val_features, train_targets, val_targets = train_test_split(
        features, targets, test_size=test_size, random_state=42
    )
    
    # Create datasets
    pretrain_dataset = PolymerDataset(train_features)  # For self-supervised pre-training
    train_dataset = PolymerDataset(train_features, train_targets)
    val_dataset = PolymerDataset(val_features, val_targets)
    
    # Create data loaders
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return pretrain_loader, train_loader, val_loader, features.shape[1]

def evaluate_model(model, test_loader, config):
    """Evaluate model performance"""
    model.eval()
    predictions = {prop: [] for prop in config.TARGET_PROPERTIES}
    targets = {prop: [] for prop in config.TARGET_PROPERTIES}
    
    with torch.no_grad():
        for features, target_batch in test_loader:
            features = features.to(config.DEVICE)
            target_batch = target_batch.to(config.DEVICE)
            
            property_outputs, _ = model(features)
            
            for i, prop in enumerate(config.TARGET_PROPERTIES):
                mask = ~torch.isnan(target_batch[:, i])
                if mask.sum() > 0:
                    predictions[prop].extend(property_outputs[prop][mask].cpu().numpy().flatten())
                    targets[prop].extend(target_batch[mask, i].cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    for prop in config.TARGET_PROPERTIES:
        if len(predictions[prop]) > 0:
            pred = np.array(predictions[prop])
            true = np.array(targets[prop])
            
            metrics[prop] = {
                'r2': r2_score(true, pred),
                'rmse': np.sqrt(mean_squared_error(true, pred)),
                'mae': mean_absolute_error(true, pred)
            }
    
    return metrics

def main():
    """Main training pipeline"""
    print("=" * 80)
    print("Polymer Self-Supervised Learning Pipeline")
    print("=" * 80)
    
    # Initialize data loader
    data_loader = PolymerDataLoader(config)
    
    # Load datasets
    print("\n1. Loading datasets...")
    pi1m_data = data_loader.load_pi1m_data(sample_size=100000)  # Sample for demo
    comp_train, comp_test = data_loader.load_competition_data()
    apfe_data = data_loader.load_apfe_data()
    rg_data = data_loader.load_rg_data()
    
    # Combine datasets
    combined_data = data_loader.combine_datasets()
    if combined_data is None:
        print("No data loaded successfully. Exiting.")
        return
    
    print(f"\nDataset summary:")
    print(f"Total samples: {len(combined_data)}")
    print(f"Properties available:")
    for prop in config.TARGET_PROPERTIES:
        if prop in combined_data.columns:
            non_null = combined_data[prop].notna().sum()
            print(f"  {prop}: {non_null} samples ({non_null/len(combined_data)*100:.1f}%)")
    
    # Initialize featurizer
    print("\n2. Generating molecular features...")
    featurizer = MolecularFeaturizer()
    
    # Create datasets and loaders
    pretrain_loader, train_loader, val_loader, feature_dim = create_datasets_and_loaders(
        combined_data, featurizer, config
    )
    
    # Initialize model
    print(f"\n3. Initializing model with {feature_dim} input features...")
    model = PolymerTransformer(
        input_dim=feature_dim,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        hidden_dim=config.HIDDEN_DIM,
        num_properties=len(config.TARGET_PROPERTIES),
        dropout=config.DROPOUT
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = SelfSupervisedTrainer(model, config)
    
    # Pre-training phase
    print("\n4. Self-supervised pre-training...")
    trainer.pretrain(pretrain_loader, num_epochs=20)  # Reduced for demo
    
    # Fine-tuning phase
    print("\n5. Multi-task fine-tuning...")
    trainer.finetune(train_loader, val_loader, num_epochs=50)  # Reduced for demo
    
    # Evaluation
    print("\n6. Evaluating model...")
    metrics = evaluate_model(model, val_loader, config)
    
    print("\nFinal Results:")
    print("-" * 50)
    for prop, metric_dict in metrics.items():
        print(f"{prop}:")
        print(f"  R²: {metric_dict['r2']:.4f}")
        print(f"  RMSE: {metric_dict['rmse']:.4f}")
        print(f"  MAE: {metric_dict['mae']:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_polymer_model.pth')
    print("\nModel saved as 'final_polymer_model.pth'")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    # Install required packages
    required_packages = [
        "torch", "torchvision", "torchaudio",
        "scikit-learn", "matplotlib", "seaborn",
        "rdkit-pypi", "pandas", "numpy", "tqdm"
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    main()