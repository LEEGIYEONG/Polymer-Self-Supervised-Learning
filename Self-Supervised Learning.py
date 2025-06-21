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
    EMBEDDING_DIM = 512  # Reduced from 512
    HIDDEN_DIM = 256     # Reduced from 256
    NUM_LAYERS = 4       # Reduced from 4
    NUM_HEADS = 8        # Reduced from 8
    DROPOUT = 0.1
    
    # Training parameters (memory optimized)
    BATCH_SIZE = 34      # Reduced from 32
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
        """Load APFEforPI thermal conductivity data with all descriptors"""
        print("Loading APFE thermal conductivity data...")
        
        try:
            # Load all APFE descriptor files
            apfe_files = {
                'example': self.config.APFE_PATH / "example_smiles.csv",
                'des_md': self.config.APFE_PATH / "Des_MD.csv",
                'des_monomer': self.config.APFE_PATH / "Des_monomer.csv", 
                'des_mordred': self.config.APFE_PATH / "Des_Mordred.csv"
            }
            
            apfe_datasets = {}
            for name, file_path in apfe_files.items():
                if file_path.exists():
                    data = pd.read_csv(file_path)
                    print(f"Loaded {name}: {data.shape}")
                    apfe_datasets[name] = data
                else:
                    print(f"File not found: {file_path}")
            
            if not apfe_datasets:
                print("No APFE files found")
                return None
            
            # Merge all datasets on common ID/SMILES columns
            combined_apfe = None
            
            # Start with monomer data (has TC values)
            if 'des_monomer' in apfe_datasets:
                combined_apfe = apfe_datasets['des_monomer'].copy()
                # Rename TC to Tc for consistency
                if 'TC' in combined_apfe.columns:
                    combined_apfe = combined_apfe.rename(columns={'TC': 'Tc'})
                print(f"Base dataset: des_monomer with {len(combined_apfe)} samples")
            
            # Merge MD descriptors
            if 'des_md' in apfe_datasets and combined_apfe is not None:
                md_data = apfe_datasets['des_md']
                # Merge on ID, exclude duplicate SMILES column
                md_cols = [col for col in md_data.columns if col not in ['SMILES', 'Smiles']]
                combined_apfe = combined_apfe.merge(
                    md_data[md_cols], 
                    on='ID', 
                    how='left', 
                    suffixes=('', '_md')
                )
                print(f"Merged MD descriptors: {combined_apfe.shape}")
            
            # Merge Mordred descriptors  
            if 'des_mordred' in apfe_datasets and combined_apfe is not None:
                mordred_data = apfe_datasets['des_mordred']
                # Clean up column names - Mordred has many descriptors
                mordred_cols = [col for col in mordred_data.columns if col not in ['SMILES', 'Smiles', 'Unnamed: 0']]
                combined_apfe = combined_apfe.merge(
                    mordred_data[mordred_cols],
                    on='ID',
                    how='left',
                    suffixes=('', '_mordred')
                )
                print(f"Merged Mordred descriptors: {combined_apfe.shape}")
            
            # If no monomer data, try example file
            if combined_apfe is None and 'example' in apfe_datasets:
                combined_apfe = apfe_datasets['example'].copy()
                if 'TC' in combined_apfe.columns:
                    combined_apfe = combined_apfe.rename(columns={'TC': 'Tc'})
                print(f"Using example file: {len(combined_apfe)} samples")
            
            if combined_apfe is not None:
                print(f"Final APFE dataset: {combined_apfe.shape}")
                print(f"Columns: {list(combined_apfe.columns)}")
                
                # Store individual datasets for reference
                self.datasets.update({f'apfe_{k}': v for k, v in apfe_datasets.items()})
                self.datasets['apfe'] = combined_apfe
                
                return combined_apfe
            else:
                print("Could not create combined APFE dataset")
                return None
            
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
                print(f"Loading {file.name}...")
                
                # Read file with proper format handling
                data = []
                with open(file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                sequence = parts[0]
                                try:
                                    rg_value = float(parts[-1])  # Last value is Rg
                                    data.append({'sequence': sequence, 'Rg': rg_value, 'q_value': q_value})
                                except ValueError:
                                    continue
                
                if data:
                    df = pd.DataFrame(data)
                    rg_data_list.append(df)
                    print(f"  Loaded {len(df)} sequences from {file.name}")
            
            if rg_data_list:
                rg_data = pd.concat(rg_data_list, ignore_index=True)
                
                # Convert sequence to SMILES-like format (simplified)
                print("Converting sequences to SMILES...")
                rg_data['SMILES'] = rg_data['sequence'].apply(self._sequence_to_smiles)
                
                # Remove invalid SMILES
                valid_mask = rg_data['SMILES'].notna()
                rg_data = rg_data[valid_mask].reset_index(drop=True)
                
                print(f"Loaded {len(rg_data)} valid Rg samples across {len(rg_data_list)} q-values")
                print(f"Q-values: {sorted(rg_data['q_value'].unique())}")
                print(f"Rg range: {rg_data['Rg'].min():.3f} - {rg_data['Rg'].max():.3f}")
                
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
        
        # Clean sequence
        sequence = str(sequence).strip()
        if not sequence or len(sequence) == 0:
            return None
        
        # For the Rg dataset, sequences are binary (1 and 2 representing different monomers)
        # This is a simplified conversion - in practice, you'd need more sophisticated mapping
        try:
            # Map sequence numbers to chemical groups
            monomer_map = {
                '1': 'C(C)',     # Monomer type 1 (e.g., propylene-like)
                '2': 'CC',       # Monomer type 2 (e.g., ethylene-like)
            }
            
            # Convert sequence to SMILES segments
            smiles_parts = []
            for char in sequence:
                if char in monomer_map:
                    smiles_parts.append(monomer_map[char])
                elif char.isdigit():
                    # For other numbers, default to simple carbon
                    smiles_parts.append('C')
            
            if smiles_parts and len(smiles_parts) > 0:
                # Join segments and add polymer notation
                # Limit length to avoid memory issues
                if len(smiles_parts) > 50:  # Truncate very long sequences
                    smiles_parts = smiles_parts[:50]
                
                # Create simplified polymer SMILES
                polymer_smiles = '*' + '-'.join(smiles_parts[:10]) + '*'  # Use first 10 units
                return polymer_smiles
            else:
                return None
                
        except Exception as e:
            print(f"Error converting sequence '{sequence[:50]}...': {e}")
            return None

    def combine_datasets(self):
        """Combine all loaded datasets"""
        print("Combining datasets...")
        
        combined_list = []
        
        # Process each dataset
        for name, data in self.datasets.items():
            if data is not None and len(data) > 0:
                print(f"Processing {name} dataset: {data.shape}")
                
                # Standardize column names
                processed_data = data.copy()
                
                # Ensure SMILES column exists
                if 'SMILES' not in processed_data.columns:
                    if 'smiles' in processed_data.columns:
                        processed_data['SMILES'] = processed_data['smiles']
                    else:
                        print(f"  Skipping {name}: No SMILES column found")
                        continue
                
                # Add dataset source
                processed_data['source'] = name
                
                # Select relevant columns
                keep_cols = ['SMILES', 'source']
                
                # Add target properties if available
                for prop in self.config.TARGET_PROPERTIES:
                    if prop in processed_data.columns:
                        keep_cols.append(prop)
                        print(f"  Found {prop}: {processed_data[prop].notna().sum()} non-null values")
                  # Add additional useful columns
                additional_cols = ['q_value', 'SA Score', 'SA_Score', 'ID']  # Include both versions of SA Score
                for col in additional_cols:
                    if col in processed_data.columns and col not in keep_cols:
                        keep_cols.append(col)
                
                # Filter columns that actually exist
                available_cols = [col for col in keep_cols if col in processed_data.columns]
                processed_data = processed_data[available_cols]
                
                print(f"  Keeping columns: {available_cols}")
                combined_list.append(processed_data)
        
        if combined_list:
            self.combined_data = pd.concat(combined_list, ignore_index=True, sort=False)
            print(f"Combined dataset shape: {self.combined_data.shape}")
            
            # Remove invalid SMILES
            print("Filtering invalid SMILES...")
            initial_count = len(self.combined_data)
            valid_smiles = self.combined_data['SMILES'].apply(self._is_valid_smiles)
            self.combined_data = self.combined_data[valid_smiles].reset_index(drop=True)
            final_count = len(self.combined_data)
            print(f"After filtering invalid SMILES: {final_count} ({initial_count - final_count} removed)")
            
            # Print data source distribution
            print("\nData source distribution:")
            source_counts = self.combined_data['source'].value_counts()
            for source, count in source_counts.items():
                print(f"  {source}: {count} samples ({count/len(self.combined_data)*100:.1f}%)")
            
            # Print property availability
            print("\nProperty availability:")
            for prop in self.config.TARGET_PROPERTIES:
                if prop in self.combined_data.columns:
                    non_null = self.combined_data[prop].notna().sum()
                    print(f"  {prop}: {non_null} samples ({non_null/len(self.combined_data)*100:.1f}%)")
                else:
                    print(f"  {prop}: 0 samples (0.0%)")
            
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
    
    def __init__(self, feature_types=['morgan', 'descriptors'], use_pi1m_sa=True, use_apfe_descriptors=True):
        self.feature_types = feature_types
        self.use_pi1m_sa = use_pi1m_sa  # Use SA scores from PI1M v2 if available
        self.use_apfe_descriptors = use_apfe_descriptors  # Use APFE MD/Mordred descriptors
        self.scaler = StandardScaler()
        
    def featurize_smiles(self, smiles_list, sa_scores=None, apfe_md_features=None, apfe_mordred_features=None):
        """Generate features for list of SMILES"""
        features_list = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Featurizing SMILES")):
            sa_score = sa_scores[i] if sa_scores is not None and i < len(sa_scores) else None
            md_features = apfe_md_features.iloc[i] if apfe_md_features is not None and i < len(apfe_md_features) else None
            mordred_features = apfe_mordred_features.iloc[i] if apfe_mordred_features is not None and i < len(apfe_mordred_features) else None
            
            features = self._get_molecular_features(smiles, sa_score, md_features, mordred_features)
            features_list.append(features)
        
        return np.array(features_list)
    
    def _get_molecular_features(self, smiles, sa_score=None, md_features=None, mordred_features=None):
        """Get molecular features for a single SMILES"""
        try:
            if pd.isna(smiles) or smiles is None or 'nan' in str(smiles).lower():
                return np.zeros(self._get_feature_dim())
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self._get_feature_dim())
            
            features = []
            if 'morgan' in self.feature_types:                
                try:
                    # Morgan fingerprints (1024 bits - reduced for memory)
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    features.extend([int(x) for x in fp.ToBitString()])
                except Exception:
                    features.extend([0] * 1024)  # Use zeros if fingerprint calculation fails
            
            if 'descriptors' in self.feature_types:
                # Molecular descriptors - using correct RDKit API
                try:
                    desc_features = [
                        Descriptors.MolWt(mol),
                        Crippen.MolLogP(mol),  # Use Crippen.MolLogP instead of Descriptors.LogP
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumAromaticRings(mol),
                        Descriptors.NumAliphaticRings(mol),
                        Descriptors.RingCount(mol),
                        Descriptors.FractionCSP3(mol),  # Corrected: FractionCSP3 not FractionCsp3
                    ]
                    features.extend(desc_features)
                except Exception as e:
                    # If any descriptor calculation fails, use zeros
                    features.extend([0.0] * 10)
                    print(f"Warning: Failed to calculate descriptors for SMILES, using zeros")
            
            if 'sa_score' in self.feature_types:
                # Use PI1M SA score if available, otherwise calculate
                if sa_score is not None and not pd.isna(sa_score):
                    features.append(float(sa_score))
                else:
                    try:
                        # Simplified SA score calculation
                        complexity = rdMolDescriptors.BertzCT(mol)
                        # Normalize to 0-1 range (empirically derived)
                        normalized_sa = 1.0 / (1.0 + np.exp(-(complexity - 100) / 50))
                        features.append(normalized_sa)
                    except:
                        features.append(0.5)  # Default middle value
            
            # Add APFE MD descriptors if available
            if 'apfe_md' in self.feature_types and md_features is not None:
                md_cols = ['Mass_max', 'Mass_min', 'Mass_ave', 'Charge_max', 'Charge_min', 'Charge_ave',
                          'Epsilon_max', 'Epsilon_min', 'Epsilon_ave', 'Sigma_max', 'Sigma_min', 'Sigma_ave',
                          'K_bond_max', 'K_bond_min', 'K_bond_ave', 'R0_max', 'R0_min', 'R0_ave',
                          'K_ang_max', 'K_ang_min', 'K_ang_ave', 'Theta0_max', 'Theta0_min', 'Theta0_ave',
                          'K_dih_max', 'K_dih_min', 'K_dih_ave']
                
                for col in md_cols:
                    if col in md_features.index:
                        val = md_features[col]
                        if pd.isna(val):
                            features.append(0.0)
                        else:
                            features.append(float(val))
                    else:
                        features.append(0.0)
            
            # Add APFE Mordred descriptors if available (select key ones to avoid memory issues)
            if 'apfe_mordred' in self.feature_types and mordred_features is not None:
                # Select important Mordred descriptors to avoid feature explosion
                key_mordred_cols = ['ABC', 'ABCGG', 'SpMax_A', 'SpMAD_A', 'LogEE_A', 'VE1_A', 'VR1_A', 'VR3_A',
                                   'nHetero', 'nH', 'nN', 'nO', 'BalabanJ', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3',
                                   'HybRatio', 'GeomShapeIndex', 'Kier3', 'nRot', 'RotRatio']
                for col in key_mordred_cols:
                    if col in mordred_features.index:
                        val = mordred_features[col]
                        if pd.isna(val):
                            features.append(0.0)
                        else:
                            features.append(float(val))
                    else:
                        features.append(0.0)

            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            # Only print specific SMILES errors occasionally to avoid spam
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            
            self._error_count += 1
            
            if self._error_count <= 5:  # Only print first 5 errors
                print(f"Error featurizing SMILES '{str(smiles)[:50]}...': {e}")
            elif self._error_count == 6:
                print("Suppressing further featurization error messages...")
            
            return np.zeros(self._get_feature_dim())
    
    def _get_feature_dim(self):
        """Get total feature dimension"""
        dim = 0
        if 'morgan' in self.feature_types:
            dim += 1024  # Reduced from 2048
        if 'descriptors' in self.feature_types:
            dim += 10
        if 'sa_score' in self.feature_types:
            dim += 1
        if 'apfe_md' in self.feature_types:
            dim += 27  # MD descriptors
        if 'apfe_mordred' in self.feature_types:
            dim += 23  # Selected key Mordred descriptors
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
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                for i, prop in enumerate(self.config.TARGET_PROPERTIES):
                    if not torch.isnan(targets[:, i]).all():  # Skip if all NaN
                        mask = ~torch.isnan(targets[:, i])
                        if mask.sum() > 0:
                            prop_loss = self.mse_loss(
                                property_outputs[prop][mask].squeeze(), 
                                targets[mask, i]
                            )
                            total_loss = total_loss + prop_loss
                            train_losses[prop].append(prop_loss.item())
                
                train_losses['total'].append(total_loss.item())
                
                # Backward pass
                if total_loss.item() > 0:  # Only backward if there's actual loss
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
                
                batch_loss = torch.tensor(0.0, device=self.device)
                for i, prop in enumerate(self.config.TARGET_PROPERTIES):
                    if not torch.isnan(targets[:, i]).all():
                        mask = ~torch.isnan(targets[:, i])
                        if mask.sum() > 0:
                            prop_loss = self.mse_loss(
                                property_outputs[prop][mask].squeeze(), 
                                targets[mask, i]
                            )
                            batch_loss = batch_loss + prop_loss
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')

def create_datasets_and_loaders(combined_data, featurizer, config, test_size=0.2):
    """Create datasets and data loaders"""
    print("Creating datasets and data loaders...")
      # Extract SA scores if available from PI1M v2
    sa_scores = None
    if 'SA Score' in combined_data.columns:  # PI1M v2 uses 'SA Score' with space
        sa_scores = combined_data['SA Score'].values
        if 'sa_score' not in featurizer.feature_types:
            featurizer.feature_types.append('sa_score')
        print("Using SA scores from PI1M v2 dataset")
    elif 'SA_Score' in combined_data.columns:  # Fallback for underscore version
        sa_scores = combined_data['SA_Score'].values
        if 'sa_score' not in featurizer.feature_types:
            featurizer.feature_types.append('sa_score')
        print("Using SA scores from dataset")
    
    # Extract APFE MD descriptors if available
    apfe_md_features = None
    if any(col.startswith('Mass_') or col.startswith('Charge_') or col.startswith('Epsilon_') 
           for col in combined_data.columns):
        md_cols = [col for col in combined_data.columns 
                   if col.startswith(('Mass_', 'Charge_', 'Epsilon_', 'Sigma_', 'K_bond_', 'R0_', 'K_ang_', 'Theta0_', 'K_dih_'))]
        if md_cols:
            apfe_md_features = combined_data[md_cols]
            if 'apfe_md' not in featurizer.feature_types:
                featurizer.feature_types.append('apfe_md')
            print(f"Using APFE MD descriptors: {len(md_cols)} features")
    
    # Extract APFE Mordred descriptors if available (subset)
    apfe_mordred_features = None
    key_mordred_cols = ['ABC', 'ABCGG', 'SpMax_A', 'SpMAD_A', 'LogEE_A', 'VE1_A', 'VR1_A', 'VR3_A',
                       'nHetero', 'nH', 'nN', 'nO', 'BalabanJ', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3',
                       'HybRatio', 'GeomShapeIndex', 'Kier3', 'nRot', 'RotRatio']
    
    available_mordred_cols = [col for col in key_mordred_cols if col in combined_data.columns]
    if available_mordred_cols:
        apfe_mordred_features = combined_data[available_mordred_cols]
        if 'apfe_mordred' not in featurizer.feature_types:
            featurizer.feature_types.append('apfe_mordred')
        print(f"Using APFE Mordred descriptors: {len(available_mordred_cols)} features")
    
    # Generate molecular features
    features = featurizer.featurize_smiles(
        combined_data['SMILES'].tolist(), 
        sa_scores, 
        apfe_md_features, 
        apfe_mordred_features
    )
    
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
      # Print dataset loading summary
    print("\nDataset loading summary:")
    print(f"  PI1M: {'✓' if pi1m_data is not None else '✗'} ({len(pi1m_data) if pi1m_data is not None else 0} samples)")
    if pi1m_data is not None:
        has_sa_score = 'SA Score' in pi1m_data.columns
        print(f"    - SA Score available: {'✓' if has_sa_score else '✗'}")
        if has_sa_score:
            print(f"    - SA Score range: {pi1m_data['SA Score'].min():.3f} - {pi1m_data['SA Score'].max():.3f}")
    print(f"  Competition: {'✓' if comp_train is not None else '✗'} ({len(comp_train) if comp_train is not None else 0} train samples)")
    print(f"  APFE: {'✓' if apfe_data is not None else '✗'} ({len(apfe_data) if apfe_data is not None else 0} samples)")
    print(f"  Rg data: {'✓' if rg_data is not None else '✗'} ({len(rg_data) if rg_data is not None else 0} samples)")
    
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
      # Initialize featurizer with APFE descriptors
    print("\n2. Generating molecular features...")
    featurizer = MolecularFeaturizer(
        feature_types=['morgan', 'descriptors'], 
        use_pi1m_sa=True, 
        use_apfe_descriptors=True
    )
    
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