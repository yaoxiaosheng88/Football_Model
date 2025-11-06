"""
Data loading module for football prediction system.
Loads all league CSVs and merges them into a master dataframe.
Now integrates API data for enhanced coverage.
"""

import pandas as pd
import os
from pathlib import Path
from utils.team_name_normalizer import normalize_team_name
from utils.feature_engineering import compute_team_season_averages


def convert_api_to_historical_format(api_df, league_code):
    """
    Convert API CSV format to historical CSV format.
    Each API match becomes 2 rows (home team perspective, away team perspective).
    
    Args:
        api_df: DataFrame from API CSV (filtered by league)
        league_code: League code (EPL, LALIGA, etc.)
        
    Returns:
        pd.DataFrame: Converted dataframe in historical format
    """
    if len(api_df) == 0:
        return pd.DataFrame()
    
    rows = []
    
    # Map API league names to league codes
    league_mapping = {
        'Premier League': 'EPL',
        'La Liga': 'LALIGA',
        'Bundesliga': 'BUNDESLIGA',
        'Serie A': 'SERIEA',
        'Ligue 1': 'LIGUE1',
        'UEFA Champions League': 'UCL'
    }
    
    for _, row in api_df.iterrows():
        # Only process finished matches with valid scores
        if row.get('status') != 'FINISHED':
            continue
        
        home_team = row.get('home_team_name', '')
        away_team = row.get('away_team_name', '')
        home_score = row.get('score_fullTime_home', '')
        away_score = row.get('score_fullTime_away', '')
        utc_date = row.get('utcDate', '')
        api_league = row.get('league', '')
        
        # Skip if missing critical data
        if pd.isna(home_team) or pd.isna(away_team) or home_team == 'nan' or away_team == 'nan':
            continue
        
        if pd.isna(home_score) or pd.isna(away_score) or home_score == 'nan' or away_score == 'nan':
            continue
        
        # Normalize team names (they should already be normalized, but ensure consistency)
        home_team_norm = normalize_team_name(home_team)
        away_team_norm = normalize_team_name(away_team)
        
        # Parse date
        try:
            date_val = pd.to_datetime(utc_date, errors='coerce').normalize()
            if pd.isna(date_val):
                continue
        except:
            continue
        
        # Convert scores to numeric
        try:
            home_score_num = float(home_score)
            away_score_num = float(away_score)
        except:
            continue
        
        # Determine Result (W/L/D)
        if home_score_num > away_score_num:
            home_result = 'W'
            away_result = 'L'
        elif home_score_num < away_score_num:
            home_result = 'L'
            away_result = 'W'
        else:
            home_result = 'D'
            away_result = 'D'
        
        # Extract season from date (2025-01-01 -> 2025)
        season_year = date_val.year
        # Determine season string (2025-2026 for 2025 season)
        # API data is for 2025 season, which spans 2025-2026
        # For 2025 dates, season is 2025-2026
        if season_year == 2025:
            season_str = "2025-2026"
            season_year_int = 2025  # Use 2025 for season weighting
        elif date_val.month >= 7:  # July onwards = current season
            season_str = f"{season_year}-{season_year + 1}"
            season_year_int = season_year
        else:  # January-June = previous season
            season_str = f"{season_year - 1}-{season_year}"
            season_year_int = season_year - 1
        
        # Use league code from mapping or fallback
        final_league_code = league_mapping.get(api_league, league_code)
        
        # Create home team perspective row
        home_row = {
            'date': date_val,
            'team': home_team_norm,
            'opponent': away_team_norm,
            'Round': None,  # Not available in API
            'Venue': 'Home',
            'Result': home_result,
            'GF': home_score_num,
            'GA': away_score_num,
            'xG': pd.NA,  # Not available in API
            'xGA': pd.NA,
            'Poss': pd.NA,
            'Season': season_str,
            'Time': None,
            'Day': None,
            'Performance_SoTA': pd.NA,
            'Standard_Sh': pd.NA,
            'Standard_SoT': pd.NA,
            'Performance_PSxG': pd.NA,
            'Crosses_Stp': pd.NA,
            'Crosses_Stp%': pd.NA,
            'league': final_league_code,
            'season_year': season_year_int
        }
        
        # Create away team perspective row
        away_row = {
            'date': date_val,
            'team': away_team_norm,
            'opponent': home_team_norm,
            'Round': None,
            'Venue': 'Away',
            'Result': away_result,
            'GF': away_score_num,
            'GA': home_score_num,
            'xG': pd.NA,
            'xGA': pd.NA,
            'Poss': pd.NA,
            'Season': season_str,
            'Time': None,
            'Day': None,
            'Performance_SoTA': pd.NA,
            'Standard_Sh': pd.NA,
            'Standard_SoT': pd.NA,
            'Performance_PSxG': pd.NA,
            'Crosses_Stp': pd.NA,
            'Crosses_Stp%': pd.NA,
            'league': final_league_code,
            'season_year': season_year_int
        }
        
        rows.append(home_row)
        rows.append(away_row)
    
    if len(rows) == 0:
        return pd.DataFrame()
    
    converted_df = pd.DataFrame(rows)
    return converted_df


def deduplicate_matches(historical_df, api_df):
    """
    Intelligently deduplicate matches between historical and API data.
    If a match exists in both, keep historical version (has advanced metrics).
    If match only exists in API, keep API version.
    
    Args:
        historical_df: DataFrame from historical CSVs
        api_df: DataFrame from API CSV (converted to historical format)
        
    Returns:
        pd.DataFrame: Merged dataframe with duplicates removed
    """
    if len(api_df) == 0:
        return historical_df.copy()
    
    if len(historical_df) == 0:
        return api_df.copy()
    
    # Create match identifiers for deduplication
    # Match key: (team, opponent, date within 1 day tolerance, league)
    # Use normalized team names and normalized dates (YYYY-MM-DD format)
    # CRITICAL: Handle NaN dates by using a date-independent key for those matches
    historical_df['_has_date'] = historical_df['date'].notna()
    api_df['_has_date'] = api_df['date'].notna()
    
    # For matches with dates: use date-based matching
    historical_df['_match_key'] = historical_df.apply(
        lambda row: (
            str(row['team']).strip() + '|' +
            str(row['opponent']).strip() + '|' +
            (row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'NO_DATE') + '|' +
            str(row['league'])
        ), axis=1
    )
    
    api_df['_match_key'] = api_df.apply(
        lambda row: (
            str(row['team']).strip() + '|' +
            str(row['opponent']).strip() + '|' +
            (row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'NO_DATE') + '|' +
            str(row['league'])
        ), axis=1
    )
    
    # Also create reverse match key (team/opponent swapped) for better matching
    # Since home/away perspective might differ
    historical_df['_match_key_reverse'] = historical_df.apply(
        lambda row: (
            str(row['opponent']).strip() + '|' +
            str(row['team']).strip() + '|' +
            (row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'NO_DATE') + '|' +
            str(row['league'])
        ), axis=1
    )
    
    api_df['_match_key_reverse'] = api_df.apply(
        lambda row: (
            str(row['opponent']).strip() + '|' +
            str(row['team']).strip() + '|' +
            (row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'NO_DATE') + '|' +
            str(row['league'])
        ), axis=1
    )
    
    # CRITICAL FIX: For La Liga, prioritize historical data over API data
    # Historical data has advanced metrics, so if a match exists in both, always keep historical
    # Only mark API matches as duplicates if they have dates AND match historical
    # For matches without dates in historical, don't deduplicate (keep both)
    
    # Find API matches that don't exist in historical (check both directions)
    # BUT: Only check against historical matches that have valid dates
    # This protects historical data without dates from being incorrectly deduplicated
    historical_with_dates = historical_df[historical_df['_has_date'] == True]
    historical_keys_with_dates = set(historical_with_dates['_match_key'].unique()) | set(historical_with_dates['_match_key_reverse'].unique())
    
    # API match is duplicate ONLY if it matches a historical match WITH a date
    # This ensures historical data without dates is preserved
    api_duplicate_mask = (
        api_df['_match_key'].isin(historical_keys_with_dates) | 
        api_df['_match_key_reverse'].isin(historical_keys_with_dates)
    )
    
    api_only_mask = ~api_duplicate_mask
    api_only_df = api_df[api_only_mask].copy()
    
    # Merge: historical + API-only matches
    # Filter out temporary columns before concat to avoid FutureWarning
    historical_df_clean = historical_df.drop(columns=['_match_key', '_match_key_reverse', '_has_date'])
    api_only_df_clean = api_only_df.drop(columns=['_match_key', '_match_key_reverse', '_has_date'])
    
    merged_df = pd.concat([historical_df_clean, api_only_df_clean], ignore_index=True)
    
    return merged_df


def load_all_league_data(data_dir='data'):
    """
    Load all league CSV files and merge into master dataframe.
    Now integrates API data for enhanced coverage (2025-2026 season).
    
    Args:
        data_dir: Directory containing league CSV files
        
    Returns:
        pd.DataFrame: Master dataframe with all league data (historical + API)
    """
    leagues = ['epl', 'laliga', 'bundesliga', 'seriea', 'ligue1', 'mls', 'ucl']
    
    # Map league codes for API integration
    league_code_mapping = {
        'epl': 'EPL',
        'laliga': 'LALIGA',
        'bundesliga': 'BUNDESLIGA',
        'seriea': 'SERIEA',
        'ligue1': 'LIGUE1',
        'ucl': 'UCL'
    }
    
    all_data = []
    
    # Load API data once
    api_path = os.path.join(data_dir, 'api_results_basic.csv')
    api_data = None
    if os.path.exists(api_path):
        try:
            api_df = pd.read_csv(api_path, dtype=str)
            # Clean dates
            if 'utcDate' in api_df.columns:
                api_df['date'] = pd.to_datetime(api_df['utcDate'], errors='coerce').dt.normalize()
            else:
                api_df['date'] = pd.NaT
            api_data = api_df
            print(f"Loaded API data: {len(api_df)} rows")
        except Exception as e:
            print(f"Warning: Could not load API data: {e}")
    
    # Process each league
    for league in leagues:
        csv_path = os.path.join(data_dir, f'{league}_final.csv')
        league_code = league.upper()
        
        # Load historical CSV
        historical_df = None
        if os.path.exists(csv_path):
            historical_df = pd.read_csv(csv_path)
            historical_df['league'] = league_code
            
            # Normalize team names for consistency
            if 'team' in historical_df.columns:
                historical_df['team'] = historical_df['team'].apply(normalize_team_name)
            if 'opponent' in historical_df.columns:
                historical_df['opponent'] = historical_df['opponent'].apply(normalize_team_name)
            
            # Standardize date column
            historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
            
            # Extract season year for weighting
            if 'Season' in historical_df.columns:
                historical_df['season_year'] = historical_df['Season'].apply(extract_season_year)
            
            print(f"Loaded {league} (historical): {len(historical_df)} rows")
        else:
            print(f"Warning: {csv_path} not found")
            historical_df = pd.DataFrame()
        
        # Load and convert API data for this league (except MLS)
        api_df_converted = pd.DataFrame()
        if api_data is not None and league.lower() != 'mls':
            # Map API league names to our league codes
            api_league_mapping = {
                'Premier League': 'EPL',
                'La Liga': 'LALIGA',
                'Bundesliga': 'BUNDESLIGA',
                'Serie A': 'SERIEA',
                'Ligue 1': 'LIGUE1',
                'UEFA Champions League': 'UCL'
            }
            
            # Filter API data by league
            api_league_name = None
            for api_name, code in api_league_mapping.items():
                if code == league_code:
                    api_league_name = api_name
                    break
            
            if api_league_name:
                api_league_df = api_data[api_data['league'] == api_league_name].copy()
                if len(api_league_df) > 0:
                    # CRITICAL: Filter API data to only include matches AFTER historical data cutoff
                    # Historical data ends at:
                    # - Bundesliga: Week 8
                    # - Serie A: Week 8
                    # - Ligue 1: Week 9
                    # - EPL: Week 9
                    # - La Liga: Week 10
                    # - UCL: Week 2
                    matchweek_cutoffs = {
                        'BUNDESLIGA': 8,
                        'SERIEA': 8,
                        'LIGUE1': 9,
                        'EPL': 9,
                        'LALIGA': 10,
                        'UCL': 2
                    }
                    
                    cutoff_matchweek = matchweek_cutoffs.get(league_code)
                    if cutoff_matchweek is not None:
                        # Convert matchday to numeric and filter
                        api_league_df['matchday'] = pd.to_numeric(api_league_df['matchday'], errors='coerce')
                        # Only include matches AFTER the cutoff matchweek
                        api_league_df = api_league_df[
                            (api_league_df['matchday'].notna()) & 
                            (api_league_df['matchday'] > cutoff_matchweek)
                        ].copy()
                        
                        if len(api_league_df) == 0:
                            print(f"  -> No API data after matchweek {cutoff_matchweek} for {league}")
                        else:
                            print(f"  -> Filtered API data: {len(api_league_df)} matches after matchweek {cutoff_matchweek}")
                    
                    # Convert filtered API data to historical format
                    if len(api_league_df) > 0:
                        api_df_converted = convert_api_to_historical_format(api_league_df, league_code)
                        print(f"Loaded {league} (API): {len(api_df_converted)} rows")
        
        # Merge historical and API data (intelligent deduplication)
        if len(historical_df) > 0 and len(api_df_converted) > 0:
            merged_df = deduplicate_matches(historical_df, api_df_converted)
            api_only_count = len(merged_df) - len(historical_df)
            print(f"Merged {league}: {len(merged_df)} rows (historical: {len(historical_df)}, API-only: {api_only_count} new)")
        elif len(historical_df) > 0:
            merged_df = historical_df.copy()
        elif len(api_df_converted) > 0:
            merged_df = api_df_converted.copy()
        else:
            merged_df = pd.DataFrame()
        
        if len(merged_df) > 0:
            all_data.append(merged_df)
    
    if not all_data:
        raise ValueError("No league data files found!")
    
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure date column is standardized
    master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')
    
    # Ensure season_year is set (extract from Season if not already set)
    if 'season_year' not in master_df.columns or master_df['season_year'].isna().any():
        if 'Season' in master_df.columns:
            # Extract season_year from Season column
            master_df['season_year'] = master_df['Season'].apply(extract_season_year)
        else:
            # Extract from date if Season column doesn't exist
            master_df['season_year'] = master_df['date'].dt.year
        
        # Fill any remaining NaN season_year values by extracting from date
        # This handles cases where Season column exists but has NaN values
        mask = master_df['season_year'].isna() & master_df['date'].notna()
        if mask.any():
            master_df.loc[mask, 'season_year'] = master_df.loc[mask, 'date'].dt.year
        
        # CRITICAL FIX: For historical data without dates or Season, assign season_year based on context
        # Use the most recent season_year from the same league/team as fallback
        # This ensures historical matches aren't excluded from strength calculations
        mask_still_na = master_df['season_year'].isna()
        if mask_still_na.any():
            for league_code in master_df['league'].unique():
                league_mask = (master_df['league'] == league_code) & mask_still_na
                if league_mask.any():
                    # Get available season years for this league
                    league_seasons = master_df[master_df['league'] == league_code]['season_year'].dropna().unique()
                    if len(league_seasons) > 0:
                        # Use the most recent season as fallback for matches without season_year
                        fallback_season = int(max(league_seasons))
                        master_df.loc[league_mask, 'season_year'] = fallback_season
                    else:
                        # If no seasons available, use year from current date (2025)
                        master_df.loc[league_mask, 'season_year'] = 2025
    
    # Ensure numeric columns are numeric
    numeric_cols = ['GF', 'GA', 'xG', 'xGA', 'Poss', 'Standard_Sh', 'Standard_SoT', 
                    'Performance_PSxG', 'Crosses_Stp', 'Crosses_Stp%']
    
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    
    print(f"\nTotal rows loaded: {len(master_df)}")
    print(f"Leagues: {sorted(master_df['league'].unique())}")
    print(f"Seasons: {sorted(master_df['season_year'].dropna().unique())}")
    
    return master_df


def extract_season_year(season_str):
    """Extract year from season string (e.g., '2022-2023' -> 2022)"""
    if pd.isna(season_str):
        return None
    
    season_str = str(season_str)
    
    # Handle format "2022-2023" or "2022/2023"
    if '-' in season_str or '/' in season_str:
        parts = season_str.replace('/', '-').split('-')
        if len(parts) > 0:
            try:
                return int(parts[0])
            except:
                return None
    
    # Handle single year
    try:
        return int(season_str)
    except:
        return None


def load_schedule(league, data_dir='data'):
    """
    Load schedule CSV/Excel for a league and convert to fixture format.
    
    Args:
        league: League name (epl, laliga, etc.)
        data_dir: Directory containing schedule files
        
    Returns:
        pd.DataFrame: Schedule dataframe with columns: home_team, away_team, date, round
    """
    # Try to find schedule file (handle various naming conventions)
    # Check common patterns: League_Schedule.xlsx, league_schedule.csv, etc.
    possible_paths = [
        os.path.join(data_dir, f'{league.title()}_Schedule.xlsx'),  # Epl_Schedule.xlsx
        os.path.join(data_dir, f'{league.capitalize()}_Schedule.xlsx'),  # Epl_Schedule.xlsx
        os.path.join(data_dir, f'{league}Schedule.xlsx'),  # eplSchedule.xlsx
        os.path.join(data_dir, f'{league.title()}Schedule.xlsx'),  # EplSchedule.xlsx
        os.path.join(data_dir, f'{league}_schedule.csv'),  # epl_schedule.csv
        os.path.join(data_dir, f'{league}_Schedule.csv'),  # epl_Schedule.csv
    ]
    
    df = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.xlsx'):
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path)
                if df is not None and len(df) > 0:
                    break
            except Exception as e:
                continue
    
    if df is None or len(df) == 0:
        print(f"Warning: No schedule file found for {league}")
        return pd.DataFrame()
    
    # Convert team-oriented format to fixture format
    # The Excel files have one row per team, need to pair them
    fixtures = []
    
    # Group by date and round to find matching teams
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'date' in df.columns:
        date_col = 'date'
    else:
        print(f"Warning: No date column found in schedule for {league}")
        return pd.DataFrame()
    
    if 'Team' in df.columns:
        team_col = 'Team'
    elif 'team' in df.columns:
        team_col = 'team'
    else:
        print(f"Warning: No team column found in schedule for {league}")
        return pd.DataFrame()
    
    if 'Opponent' in df.columns:
        opponent_col = 'Opponent'
    elif 'opponent' in df.columns:
        opponent_col = 'opponent'
    else:
        print(f"Warning: No opponent column found in schedule for {league}")
        return pd.DataFrame()
    
    if 'Venue' in df.columns:
        venue_col = 'Venue'
    elif 'venue' in df.columns:
        venue_col = 'venue'
    else:
        venue_col = None
    
    if 'Round' in df.columns:
        round_col = 'Round'
    elif 'round' in df.columns:
        round_col = 'round'
    else:
        round_col = None
    
    # Parse dates with multiple format support (YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY, etc.)
    # This handles mixed date formats in Excel/CSV files
    def parse_date_flexible(date_val):
        """Parse date from various formats and normalize to datetime."""
        if pd.isna(date_val):
            return pd.NaT
        
        # If already a datetime/Timestamp, return normalized
        if isinstance(date_val, pd.Timestamp):
            return date_val.normalize()
        
        # If it's a number (Excel serial date), pandas will handle it
        if isinstance(date_val, (int, float)):
            try:
                return pd.to_datetime(date_val, origin='1899-12-30', unit='D').normalize()
            except:
                pass
        
        date_str = str(date_val).strip()
        if not date_str or date_str.lower() in ['nan', 'none', '']:
            return pd.NaT
        
        # Try multiple date formats in order of likelihood
        # First try DD-MM-YYYY (common in European formats)
        date_formats = [
            '%d-%m-%Y',      # DD-MM-YYYY (e.g., 01-12-2025)
            '%d/%m/%Y',      # DD/MM/YYYY (e.g., 01/12/2025)
            '%d.%m.%Y',      # DD.MM.YYYY (e.g., 01.12.2025)
            '%Y-%m-%d',      # YYYY-MM-DD (e.g., 2025-12-01)
            '%Y/%m/%d',      # YYYY/MM/DD (e.g., 2025/12/01)
            '%Y.%m.%d',      # YYYY.MM.DD (e.g., 2025.12.01)
            '%d-%m-%y',      # DD-MM-YY (e.g., 01-12-25)
            '%d/%m/%y',      # DD/MM/YY (e.g., 01/12/25)
            '%Y-%m-%d %H:%M:%S',  # With time YYYY-MM-DD
            '%d-%m-%Y %H:%M:%S',  # With time DD-MM-YYYY
            '%d/%m/%Y %H:%M:%S',  # With time DD/MM/YYYY
        ]
        
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(date_str, format=fmt, errors='raise')
                return parsed.normalize()  # Remove time component
            except (ValueError, TypeError):
                continue
        
        # Last resort: use pandas auto-parsing with dayfirst=True to prefer DD-MM-YYYY
        try:
            parsed = pd.to_datetime(date_str, dayfirst=True, errors='raise')
            return parsed.normalize()
        except:
            return pd.NaT
    
    # Apply flexible date parsing to all date values
    df[date_col] = df[date_col].apply(parse_date_flexible)
    
    # Group by ROUND first (more reliable), then by date if round is not available
    # Since each match appears twice (once per team), we need to deduplicate
    seen_matches = set()
    
    # For MLS, group by date to ensure we capture all matches properly
    # For other leagues, group by round if available
    if league.lower() == 'mls':
        group_by_col = date_col
    elif round_col and round_col in df.columns:
        group_by_col = round_col
    else:
        group_by_col = date_col
    
    for group_key, group in df.groupby(group_by_col):
        if pd.isna(group_key):
            continue
        
        # Process each row and deduplicate matches within this round/date
        for _, row in group.iterrows():
            team = str(row[team_col]).strip()
            opponent = str(row[opponent_col]).strip()
            venue = str(row[venue_col]).strip() if venue_col and venue_col in row and pd.notna(row[venue_col]) else None
            round_val = row[round_col] if round_col and round_col in row and pd.notna(row[round_col]) else None
            date_val = row[date_col] if pd.notna(row[date_col]) else None
            
            # Skip empty teams
            if not team or not opponent:
                continue
            
            # Normalize team names FIRST before deduplication
            # This ensures variants like "Nottingham Forest" and "Nott'ham Forest" are treated as the same
            team_normalized = normalize_team_name(team)
            opponent_normalized = normalize_team_name(opponent)
            
            # Create unique match identifier
            # For MLS, teams play each other twice (home and away), so we need to use date
            # to distinguish these matches, not just team pair
            # For other leagues, round is still important for deduplication
            if league.lower() == 'mls':
                # MLS: Use (team_pair, date) to allow both home and away matches to be captured
                # The venue will determine which team is home/away
                match_key = (tuple(sorted([team_normalized, opponent_normalized])), date_val)
            else:
                # Other leagues: Use (team_pair, round) as before
                match_key = (tuple(sorted([team_normalized, opponent_normalized])), round_val)
            
            # Skip if we've already processed this match (same teams, same date/round)
            if match_key in seen_matches:
                continue
            
            seen_matches.add(match_key)
            
            # Use normalized names
            team = team_normalized
            opponent = opponent_normalized
            
            # Determine home/away based on venue
            if venue:
                venue_upper = venue.upper()
                if 'HOME' in venue_upper or venue_upper == 'H':
                    home_team = team
                    away_team = opponent
                elif 'AWAY' in venue_upper or venue_upper == 'A':
                    home_team = opponent
                    away_team = team
                else:
                    # If venue is unclear, use first team as home
                    home_team = team
                    away_team = opponent
            else:
                # No venue info - assume first team is home
                home_team = team
                away_team = opponent
            
            fixtures.append({
                'date': date_val,
                'home_team': home_team,
                'away_team': away_team,
                'round': round_val if round_val else None
            })
    
    fixtures_df = pd.DataFrame(fixtures)
    # Ensure datetime and strip time component so app uses only calendar dates
    if 'date' in fixtures_df.columns:
        fixtures_df['date'] = pd.to_datetime(fixtures_df['date'], errors='coerce').dt.normalize()
    fixtures_df = fixtures_df.sort_values('date').reset_index(drop=True)
    
    return fixtures_df


def get_league_mapping():
    """Return mapping of league codes to display names"""
    return {
        'EPL': 'Premier League',
        'LALIGA': 'La Liga',
        'BUNDESLIGA': 'Bundesliga',
        'SERIEA': 'Serie A',
        'LIGUE1': 'Ligue 1',
        'MLS': 'MLS',
        'UCL': 'UEFA Champions League'
    }


def get_team_prior_from_other_leagues(team_name, all_leagues_data, current_season=None):
    """
    Check if a team exists in other domestic leagues and return its average stats.
    Used for UCL teams that might have domestic league data.
    
    Args:
        team_name: Normalized team name
        all_leagues_data: Master dataframe with all league data
        current_season: Most recent season (optional)
        
    Returns:
        dict: Team average stats from domestic league, or None if not found
    """
    # Domestic leagues (exclude UCL)
    domestic_leagues = ['EPL', 'LALIGA', 'BUNDESLIGA', 'SERIEA', 'LIGUE1', 'MLS']
    
    for league in domestic_leagues:
        league_df = all_leagues_data[all_leagues_data['league'] == league]
        if len(league_df) == 0:
            continue
        
        # Check if team exists in this league
        team_df = league_df[league_df['team'] == team_name]
        if len(team_df) > 0:
            # Found team in domestic league, compute averages
            from utils.weighting import get_current_season
            
            if current_season is None:
                current_season = get_current_season(all_leagues_data)
            
            # Get both home and away stats
            home_stats = compute_team_season_averages(
                all_leagues_data, team_name, venue='Home', current_season=current_season
            )
            away_stats = compute_team_season_averages(
                all_leagues_data, team_name, venue='Away', current_season=current_season
            )
            
            if home_stats and away_stats:
                return {
                    'league': league,
                    'home_stats': home_stats,
                    'away_stats': away_stats
                }
    
    return None

