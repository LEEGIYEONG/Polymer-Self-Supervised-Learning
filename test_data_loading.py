"""
Quick test for data loading functionality
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
class Config:
    BASE_PATH = Path(r"c:\Users\LEEGIYEONG\Downloads\ips")
    PI1M_PATH = BASE_PATH / "PI1M-master"
    COMPETITION_PATH = BASE_PATH / "neurips-open-polymer-prediction-2025 (1)"
    APFE_PATH = BASE_PATH / "APFEforPI-main" / "dataset"
    RG_PATH = BASE_PATH / "Sequence-Radius-of-gyration-Rg-data-of-a-copolymer-main"
    TARGET_PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

print("Testing data loading...")

# Test PI1M v2
print("\n1. PI1M v2 data:")
try:
    pi1m_file = Config.PI1M_PATH / "PI1M_v2.csv"
    chunk = pd.read_csv(pi1m_file, nrows=1000)
    print(f"   ✓ Loaded {len(chunk)} samples")
    print(f"   ✓ Columns: {list(chunk.columns)}")
    print(f"   ✓ SA Score available: {'SA Score' in chunk.columns}")
    if 'SA Score' in chunk.columns:
        print(f"   ✓ SA Score range: {chunk['SA Score'].min():.3f} - {chunk['SA Score'].max():.3f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test APFE data
print("\n2. APFE data:")
try:
    des_monomer_file = Config.APFE_PATH / "Des_monomer.csv"
    if des_monomer_file.exists():
        df = pd.read_csv(des_monomer_file)
        print(f"   ✓ Des_monomer.csv: {df.shape}")
        print(f"   ✓ Has TC: {'TC' in df.columns}")
    
    des_md_file = Config.APFE_PATH / "Des_MD.csv"
    if des_md_file.exists():
        df = pd.read_csv(des_md_file)
        print(f"   ✓ Des_MD.csv: {df.shape}")
    
    des_mordred_file = Config.APFE_PATH / "Des_Mordred.csv"
    if des_mordred_file.exists():
        df = pd.read_csv(des_mordred_file)
        print(f"   ✓ Des_Mordred.csv: {df.shape}")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test Competition data
print("\n3. Competition data:")
try:
    train_file = Config.COMPETITION_PATH / "train.csv"
    if train_file.exists():
        df = pd.read_csv(train_file)
        print(f"   ✓ train.csv: {df.shape}")
        print(f"   ✓ Properties: {[col for col in Config.TARGET_PROPERTIES if col in df.columns]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test Rg data
print("\n4. Rg data:")
try:
    rg_files = list(Config.RG_PATH.glob("q_*.txt"))
    print(f"   ✓ Found {len(rg_files)} q_*.txt files")
    for file in rg_files:
        with open(file, 'r') as f:
            lines = f.readlines()[:5]  # First 5 lines
        print(f"   ✓ {file.name}: {len(lines)} sample lines")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nAll data sources checked!")
