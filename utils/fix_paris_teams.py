"""
Fix Paris FC and Paris Saint-Germain team assignments in CSV.
"""

import pandas as pd

def fix_paris_teams():
    df = pd.read_csv('data/api_results_basic.csv', dtype=str)
    
    print('Fixing Paris team assignments...')
    print('='*70)
    
    # Count issues before fix
    id1045_home_wrong = len(df[(df['home_team_id']=='1045.0') & (df['home_team_name']!='Paris FC')])
    id1045_away_wrong = len(df[(df['away_team_id']=='1045.0') & (df['away_team_name']!='Paris FC')])
    id524_home_wrong = len(df[(df['home_team_id']=='524.0') & (df['home_team_name']!='Paris Saint-Germain')])
    id524_away_wrong = len(df[(df['away_team_id']=='524.0') & (df['away_team_name']!='Paris Saint-Germain')])
    
    print(f'\nBefore fix:')
    print(f'  ID 1045 (home) wrong: {id1045_home_wrong}')
    print(f'  ID 1045 (away) wrong: {id1045_away_wrong}')
    print(f'  ID 524 (home) wrong: {id524_home_wrong}')
    print(f'  ID 524 (away) wrong: {id524_away_wrong}')
    
    # Fix all instances
    df.loc[df['home_team_id']=='1045.0', 'home_team_name'] = 'Paris FC'
    df.loc[df['away_team_id']=='1045.0', 'away_team_name'] = 'Paris FC'
    df.loc[df['home_team_id']=='524.0', 'home_team_name'] = 'Paris Saint-Germain'
    df.loc[df['away_team_id']=='524.0', 'away_team_name'] = 'Paris Saint-Germain'
    
    # Verify after fix
    id1045_home_wrong_after = len(df[(df['home_team_id']=='1045.0') & (df['home_team_name']!='Paris FC')])
    id1045_away_wrong_after = len(df[(df['away_team_id']=='1045.0') & (df['away_team_name']!='Paris FC')])
    id524_home_wrong_after = len(df[(df['home_team_id']=='524.0') & (df['home_team_name']!='Paris Saint-Germain')])
    id524_away_wrong_after = len(df[(df['away_team_id']=='524.0') & (df['away_team_name']!='Paris Saint-Germain')])
    
    print(f'\nAfter fix:')
    print(f'  ID 1045 (home) wrong: {id1045_home_wrong_after}')
    print(f'  ID 1045 (away) wrong: {id1045_away_wrong_after}')
    print(f'  ID 524 (home) wrong: {id524_home_wrong_after}')
    print(f'  ID 524 (away) wrong: {id524_away_wrong_after}')
    
    # Save
    df.to_csv('data/api_results_basic.csv', index=False)
    
    if (id1045_home_wrong_after == 0 and id1045_away_wrong_after == 0 and 
        id524_home_wrong_after == 0 and id524_away_wrong_after == 0):
        print('\n[OK] All Paris team assignments fixed!')
        return True
    else:
        print('\n[WARNING] Some issues remain')
        return False

if __name__ == "__main__":
    fix_paris_teams()

