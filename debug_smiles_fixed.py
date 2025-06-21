#!/usr/bin/env python
"""
Debug script to identify problematic SMILES in the datasets
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

def test_smiles_descriptors(smiles):
    """Test descriptor calculation for a single SMILES"""
    try:
        if pd.isna(smiles) or smiles is None or 'nan' in str(smiles).lower():
            return False, "NaN or None SMILES"
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES - RDKit parsing failed"
        
        # Test descriptor calculations
        try:
            desc_features = [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCSP3(mol),  # Corrected: FractionCSP3 not FractionCsp3
            ]
            
            # Check for NaN or inf values
            for i, val in enumerate(desc_features):
                if pd.isna(val) or np.isinf(val):
                    return False, f"Descriptor {i} returned NaN/inf: {val}"
            
            return True, "OK"
        except Exception as e:
            return False, f"Descriptor calculation error: {str(e)}"
            
    except Exception as e:
        return False, f"General error: {str(e)}"

def analyze_dataset_smiles():
    """Analyze SMILES from all datasets"""
    print("=== Analyzing SMILES from all datasets ===\n")
    
    # Load datasets
    datasets = {}
    
    # PI1M dataset
    try:
        pi1m_path = r"c:\Users\LEEGIYEONG\Downloads\ips\PI1M-master\PI1M_v2.csv"
        pi1m_df = pd.read_csv(pi1m_path)
        datasets['PI1M'] = pi1m_df['SMILES'].tolist()
        print(f"PI1M: {len(datasets['PI1M'])} SMILES loaded")
    except Exception as e:
        print(f"Error loading PI1M: {e}")
    
    # Competition dataset
    try:
        comp_path = r"c:\Users\LEEGIYEONG\Downloads\ips\neurips-open-polymer-prediction-2025 (1)\train.csv"
        comp_df = pd.read_csv(comp_path)
        datasets['Competition'] = comp_df['SMILES'].tolist()
        print(f"Competition: {len(datasets['Competition'])} SMILES loaded")
    except Exception as e:
        print(f"Error loading Competition: {e}")
    
    # APFE datasets
    try:
        apfe_md_path = r"c:\Users\LEEGIYEONG\Downloads\ips\APFEforPI-main\dataset\Des_MD.csv"
        apfe_md_df = pd.read_csv(apfe_md_path)
        datasets['APFE_MD'] = apfe_md_df['SMILES'].tolist()
        print(f"APFE MD: {len(datasets['APFE_MD'])} SMILES loaded")
    except Exception as e:
        print(f"Error loading APFE MD: {e}")
    
    try:
        apfe_monomer_path = r"c:\Users\LEEGIYEONG\Downloads\ips\APFEforPI-main\dataset\Des_monomer.csv"
        apfe_monomer_df = pd.read_csv(apfe_monomer_path)
        datasets['APFE_Monomer'] = apfe_monomer_df['SMILES'].tolist()
        print(f"APFE Monomer: {len(datasets['APFE_Monomer'])} SMILES loaded")
    except Exception as e:
        print(f"Error loading APFE Monomer: {e}")
    
    try:
        apfe_mordred_path = r"c:\Users\LEEGIYEONG\Downloads\ips\APFEforPI-main\dataset\Des_Mordred.csv"
        apfe_mordred_df = pd.read_csv(apfe_mordred_path)
        datasets['APFE_Mordred'] = apfe_mordred_df['SMILES'].tolist()
        print(f"APFE Mordred: {len(datasets['APFE_Mordred'])} SMILES loaded")
    except Exception as e:
        print(f"Error loading APFE Mordred: {e}")
    
    print("\n=== Testing SMILES for descriptor calculation ===\n")
    
    # Test each dataset
    for dataset_name, smiles_list in datasets.items():
        print(f"\n--- Testing {dataset_name} ---")
        
        total_smiles = len(smiles_list)
        problematic_smiles = []
        error_types = {}
        
        # Sample test (first 100 SMILES)
        test_size = min(100, total_smiles)
        sample_smiles = smiles_list[:test_size]
        
        for i, smiles in enumerate(sample_smiles):
            success, error_msg = test_smiles_descriptors(smiles)
            if not success:
                problematic_smiles.append((i, smiles, error_msg))
                error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print(f"Tested {test_size} SMILES from {dataset_name}")
        print(f"Problematic: {len(problematic_smiles)} ({len(problematic_smiles)/test_size*100:.1f}%)")
        
        if error_types:
            print("Error types:")
            for error_type, count in error_types.items():
                print(f"  - {error_type}: {count}")
        
        # Show first few problematic SMILES
        if problematic_smiles:
            print("\nFirst few problematic SMILES:")
            for i, (idx, smiles, error_msg) in enumerate(problematic_smiles[:5]):
                smiles_str = str(smiles)[:50] + "..." if len(str(smiles)) > 50 else str(smiles)
                print(f"  {idx}: {smiles_str} -> {error_msg}")

if __name__ == "__main__":
    analyze_dataset_smiles()
