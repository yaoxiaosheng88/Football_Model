"""
Convert API CSV format to match training data format and ensure team names are normalized.
This allows API data to be safely merged with the 14 historical CSVs for training.
"""

import pandas as pd
import os
from utils.team_name_normalizer import normalize_team_name

def convert_api_csv_to_training_format():
    """
    Convert API CSV to match the format of historical training CSVs.
    Maps: home_team_name -> team, away_team_name -> opponent
    Ensures all team names are normalized using normalize_team_name.
    """
    api_path = "data/api_results_basic.csv"
    output_path = "data/api_training_format.csv"
    
    if not os.path.exists(api_path):
        print(f"[ERROR] {api_path} not found!")
        return None
    
    print("Converting API CSV to training format...")
    print("=" * 60)
    
    df = pd.read_csv(api_path, dtype=str)
    print(f"Loaded {len(df)} rows from {api_path}")
    
    # Filter out rows with missing team names
    before = len(df)
    df = df[
        (df['home_team_name'].notna()) & 
        (df['home_team_name'] != 'nan') &
        (df['away_team_name'].notna()) & 
        (df['away_team_name'] != 'nan')
    ].copy()
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with missing team names")
    
    # Normalize team names
    print("\nNormalizing team names...")
    df['team'] = df['home_team_name'].apply(normalize_team_name)
    df['opponent'] = df['away_team_name'].apply(normalize_team_name)
    
    # Verify normalization
    mismatches = []
    for idx, row in df.head(100).iterrows():
        h_raw = str(row['home_team_name']).strip()
        a_raw = str(row['away_team_name']).strip()
        h_norm = normalize_team_name(h_raw)
        a_norm = normalize_team_name(a_raw)
        
        if h_raw != h_norm or a_raw != a_norm:
            mismatches.append((idx, h_raw, h_norm, a_raw, a_norm))
    
    if mismatches:
        print(f"\n[WARNING] Found {len(mismatches)} potential normalization issues:")
        for idx, h_raw, h_norm, a_raw, a_norm in mismatches[:5]:
            print(f"  Row {idx}: '{h_raw}' -> '{h_norm}', '{a_raw}' -> '{a_norm}'")
    else:
        print("[OK] All team names normalized correctly")
    
    # Map columns to match training format
    # Keep essential columns that match historical CSV structure
    training_df = pd.DataFrame({
        'team': df['team'],
        'opponent': df['opponent'],
        'date': pd.to_datetime(df['utcDate'], errors='coerce'),
        'league': df['league'].str.upper(),
        'home_score': pd.to_numeric(df['score_fullTime_home'], errors='coerce'),
        'away_score': pd.to_numeric(df['score_fullTime_away'], errors='coerce'),
        'status': df['status'],
        'matchday': pd.to_numeric(df['matchday'], errors='coerce'),
    })
    
    # Add dummy Season column (you may want to extract from date or set manually)
    training_df['Season'] = training_df['date'].dt.year.astype(str)
    
    # Save
    training_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved {len(training_df)} rows to {output_path}")
    print(f"Columns: {list(training_df.columns)}")
    
    # Show sample
    print("\nSample of converted data:")
    print(training_df[['team', 'opponent', 'date', 'league']].head(10))
    
    return training_df

if __name__ == "__main__":
    convert_api_csv_to_training_format()

