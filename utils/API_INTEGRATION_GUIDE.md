"""
SUMMARY: API CSV Integration with Training Data

ISSUE:
- You have 14 historical CSVs with team names normalized via team_name_normalizer.py
- API CSV has different team names that might not match
- If merged directly, team names would be inconsistent → bad predictions

SOLUTION IMPLEMENTED:
1. ✅ API CSV normalized using same normalizer as historical CSVs
2. ✅ All API team variants added to normalizer dictionary
3. ✅ team_name_map.json contains normalized names
4. ✅ API CSV converted to training format (team/opponent columns)

HOW TO USE API DATA IN TRAINING:

Option 1: Load API CSV directly (RECOMMENDED)
---------------------------------------------
In your data_loader.py or training script, add:

    import pandas as pd
    from utils.team_name_normalizer import normalize_team_name
    
    # Load API CSV
    api_df = pd.read_csv('data/api_results_basic.csv')
    
    # Normalize team names (same as historical CSVs)
    api_df['team'] = api_df['home_team_name'].apply(normalize_team_name)
    api_df['opponent'] = api_df['away_team_name'].apply(normalize_team_name)
    
    # Merge with historical data
    all_data = pd.concat([historical_data, api_df], ignore_index=True)

Option 2: Use pre-converted API CSV
------------------------------------
    api_df = pd.read_csv('data/api_training_format.csv')
    # Already has 'team' and 'opponent' columns with normalized names
    all_data = pd.concat([historical_data, api_df], ignore_index=True)

VERIFICATION:
- Run: python -m utils.ensure_api_consistency
- This ensures all API teams normalize consistently

RESULT:
✅ All team names now use the same canonical format across:
   - 14 historical CSVs
   - API CSV
   - Any future API data

✅ No name mismatches → Model will work correctly!
"""

print(__doc__)

