"""
Verify that API CSV team names match the normalizer output.
This ensures API data can be safely merged with training data.
"""

import pandas as pd
from utils.team_name_normalizer import normalize_team_name

def verify_api_csv_normalization():
    """Check if API CSV team names are properly normalized."""
    csv_path = "data/api_results_basic.csv"
    
    print("Verifying API CSV normalization...")
    print("=" * 60)
    
    df = pd.read_csv(csv_path, dtype=str)
    print(f"Loaded {len(df)} rows from {csv_path}\n")
    
    mismatches = []
    for idx, row in df.iterrows():
        home_raw = str(row['home_team_name']).strip()
        away_raw = str(row['away_team_name']).strip()
        
        home_normalized = normalize_team_name(home_raw)
        away_normalized = normalize_team_name(away_raw)
        
        if home_raw != home_normalized:
            mismatches.append(('home', idx, home_raw, home_normalized))
        if away_raw != away_normalized:
            mismatches.append(('away', idx, away_raw, away_normalized))
    
    print(f"Total mismatches found: {len(mismatches)}")
    
    if mismatches:
        print("\nFirst 20 mismatches:")
        for col, idx, raw, norm in mismatches[:20]:
            print(f"  Row {idx}, {col}: '{raw}' -> '{norm}'")
        
        print("\n[WARNING] API CSV contains unnormalized team names!")
        print("Run: python -m utils.football_data_api renormalize")
        return False
    else:
        print("\n[OK] All team names in API CSV are properly normalized!")
        print("API CSV is safe to merge with training data.")
        return True

if __name__ == "__main__":
    verify_api_csv_normalization()

