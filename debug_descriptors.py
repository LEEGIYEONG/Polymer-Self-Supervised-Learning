import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

# Test descriptor calculation on some sample SMILES
test_smiles = [
    "CC(C)C",  # Simple alkane
    "CCO",     # Ethanol
    "C1=CC=CC=C1",  # Benzene
    "*CC*",    # Polymer notation
    "*-CC-*",  # Polymer with dashes
    "C(C)(C)C(C)(C)",  # Complex polymer
    "",        # Empty string
    "invalid_smiles",  # Invalid SMILES
]

def test_descriptor_calculation(smiles):
    print(f"Testing SMILES: '{smiles}'")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  Could not parse SMILES")
            return False
        
        # Test each descriptor individually
        descriptors = {
            'MolWt': Descriptors.MolWt,
            'MolLogP': Crippen.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'TPSA': Descriptors.TPSA,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'NumAliphaticRings': Descriptors.NumAliphaticRings,
            'RingCount': Descriptors.RingCount,
            'FractionCsp3': Descriptors.FractionCsp3,
        }
        
        for desc_name, desc_func in descriptors.items():
            try:
                value = desc_func(mol)
                print(f"  {desc_name}: {value}")
            except Exception as e:
                print(f"  ERROR with {desc_name}: {e}")
                return False
        
        print(f"  ✓ All descriptors calculated successfully")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

print("Testing RDKit descriptor calculation...")
for smiles in test_smiles:
    test_descriptor_calculation(smiles)
    print()
