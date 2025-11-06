"""
Streamlit web app for football prediction system.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_all_league_data, load_schedule, get_league_mapping
from model.predictor import predict_match
from utils.weighting import get_current_season
from utils.team_name_normalizer import normalize_team_name


# Page config
st.set_page_config(
    page_title="Football Prediction System",
    page_icon="⚽",
    layout="wide"
)

# Title and Refresh Button
col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.title("Football Prediction System")
with col_refresh:
    st.write("")  # Spacing
    if st.button("🔄 Refresh Fixtures", type="primary", use_container_width=True):
        # Run API refresh
        from utils.football_data_api import refresh_fixtures_for_streamlit
        from utils.metrics_calculator import compute_team_strengths
        from utils.strengths import save_all_strengths_to_csv
        
        with st.spinner("🔄 Fetching latest fixtures and updating data..."):
            success, message, details = refresh_fixtures_for_streamlit()
            
            if success:
                # Clear all caches
                st.cache_data.clear()
                
                # Reload master data
                try:
                    st.session_state.master_data = load_all_league_data('data')
                    st.session_state.data_loaded = True
                    
                    # Recompute team strengths
                    current_season = get_current_season(st.session_state.master_data)
                    leagues = ['EPL', 'LALIGA', 'BUNDESLIGA', 'SERIEA', 'LIGUE1', 'MLS', 'UCL']
                    
                    all_strengths = {}
                    for league in leagues:
                        try:
                            strengths = compute_team_strengths(st.session_state.master_data, league, current_season)
                            if strengths:
                                all_strengths[league] = strengths
                        except Exception as e:
                            print(f"Warning: Could not compute strengths for {league}: {e}")
                    
                    # Save all strengths
                    if all_strengths:
                        save_all_strengths_to_csv(all_strengths)
                    
                    st.session_state.strengths_initialized = True
                    st.session_state['force_recalculate'] = True  # Force prediction recalculation
                    
                    st.success(message)
                    st.rerun()  # Refresh the page
                except Exception as e:
                    st.error(f"✅ API data fetched, but error reloading data: {e}")
                    st.info("Please refresh the page manually.")
            else:
                st.error(message)
                if 'error' in details:
                    st.error(f"Details: {details['error']}")

# Initialize session state
if 'master_data' not in st.session_state:
    with st.spinner("Loading league data..."):
        try:
            st.session_state.master_data = load_all_league_data('data')
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.session_state.data_loaded = False
            st.stop()

# Initialize team strengths (compute on first load)
if 'strengths_initialized' not in st.session_state and st.session_state.get('data_loaded', False):
    from utils.metrics_calculator import compute_team_strengths
    from utils.weighting import get_current_season
    
    current_season = get_current_season(st.session_state.master_data)
    leagues = ['EPL', 'LALIGA', 'BUNDESLIGA', 'SERIEA', 'LIGUE1', 'MLS', 'UCL']
    
    with st.spinner("Computing team strengths for all leagues (this may take a moment)..."):
        all_strengths = {}
        for league in leagues:
            try:
                strengths = compute_team_strengths(st.session_state.master_data, league, current_season)
                if strengths:
                    all_strengths[league] = strengths
            except Exception as e:
                print(f"Warning: Could not compute strengths for {league}: {e}")
        
        # Save all strengths to master CSV
        if all_strengths:
            from utils.strengths import save_all_strengths_to_csv
            save_all_strengths_to_csv(all_strengths)
    
    st.session_state.strengths_initialized = True


@st.cache_data
def get_all_ucl_fixtures():
    """Get ALL UCL fixtures (not divided by matchweek).
    Returns all matches sorted by date with deduplication.
    """
    schedule = load_schedule('ucl', 'data')
    if len(schedule) == 0:
        return pd.DataFrame()
    
    df = schedule.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Sort by date and deduplicate
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first')
    
    return df


@st.cache_data
def get_mls_round_types():
    """Get available round types for MLS (Regular Season, Round One, Wild Card Round)."""
    schedule = load_schedule('mls', 'data')
    if len(schedule) == 0:
        return []
    
    # Get unique round types
    round_types = schedule['round'].dropna().unique().tolist()
    # Filter and standardize round type names
    valid_rounds = []
    for rt in round_types:
        rt_str = str(rt).strip()
        if 'Regular Season' in rt_str:
            valid_rounds.append('Regular Season')
        elif 'Round One' in rt_str or 'Round 1' in rt_str:
            valid_rounds.append('Round One')
        elif 'Wild Card' in rt_str:
            valid_rounds.append('Wild Card Round')
    
    # Remove duplicates while preserving order
    seen = set()
    unique_rounds = []
    for rt in valid_rounds:
        if rt not in seen:
            seen.add(rt)
            unique_rounds.append(rt)
    
    return unique_rounds


@st.cache_data
def get_mls_fixtures_for_round_type(round_type):
    """Get ALL MLS fixtures for a specific round type.
    Returns all matches sorted by date with deduplication.
    """
    schedule = load_schedule('mls', 'data')
    if len(schedule) == 0:
        return pd.DataFrame()
    
    # Filter by round type
    if round_type == 'Regular Season':
        df = schedule[schedule['round'].str.contains('Regular Season', case=False, na=False)].copy()
    elif round_type == 'Round One':
        df = schedule[schedule['round'].str.contains('Round One|Round 1', case=False, na=False, regex=True)].copy()
    elif round_type == 'Wild Card Round':
        df = schedule[schedule['round'].str.contains('Wild Card', case=False, na=False)].copy()
    else:
        return pd.DataFrame()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Ensure dates are datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Sort by date and deduplicate
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first')
    
    return df


@st.cache_data
def load_api_data():
    """Load API data from CSV and clean dates."""
    api_path = 'data/api_results_basic.csv'
    if not os.path.exists(api_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(api_path, dtype=str)
        
        # Clean dates: remove time component from utcDate
        if 'utcDate' in df.columns:
            df['date'] = pd.to_datetime(df['utcDate'], errors='coerce').dt.normalize()
        else:
            df['date'] = pd.NaT
        
        # Map API league names to our league codes
        league_mapping = {
            'Premier League': 'EPL',
            'La Liga': 'LALIGA',
            'Bundesliga': 'BUNDESLIGA',
            'Serie A': 'SERIEA',
            'Ligue 1': 'LIGUE1',
            'UEFA Champions League': 'UCL'
        }
        
        # Add normalized league code column
        df['league_code'] = df['league'].map(league_mapping)
        
        # Convert matchday to numeric (handle NaN)
        df['matchday'] = pd.to_numeric(df['matchday'], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error loading API data: {e}")
        return pd.DataFrame()


@st.cache_data
def get_fixtures_for_matchweek(league, matchweek):
    """Get fixtures for a specific league and matchweek.
    Uses API data for top 5 leagues + UCL (EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL).
    MLS uses schedule data (handled separately).
    """
    league_lower = league.lower()
    
    # Top 5 leagues + UCL: Use API data
    api_leagues = ['epl', 'laliga', 'seriea', 'bundesliga', 'ligue1', 'ucl']
    
    if league_lower in api_leagues:
        # Load API data
        api_df = load_api_data()
        
        if len(api_df) == 0:
            return pd.DataFrame()
        
        # Map league code
        league_code_mapping = {
            'epl': 'EPL',
            'laliga': 'LALIGA',
            'seriea': 'SERIEA',
            'bundesliga': 'BUNDESLIGA',
            'ligue1': 'LIGUE1',
            'ucl': 'UCL'
        }
        
        league_code = league_code_mapping.get(league_lower)
        if not league_code:
            return pd.DataFrame()
        
        # Filter by league and matchweek (convert matchweek to float for comparison)
        filtered = api_df[
            (api_df['league_code'] == league_code) & 
            (pd.to_numeric(api_df['matchday'], errors='coerce') == float(matchweek))
        ].copy()
        
        if len(filtered) == 0:
            return pd.DataFrame()
        
        # Filter out rows with missing team names (unscheduled matches)
        filtered = filtered[
            (filtered['home_team_name'].notna()) & 
            (filtered['home_team_name'] != 'nan') &
            (filtered['home_team_name'] != '') &
            (filtered['away_team_name'].notna()) & 
            (filtered['away_team_name'] != 'nan') &
            (filtered['away_team_name'] != '')
        ].copy()
        
        if len(filtered) == 0:
            return pd.DataFrame()
        
        # Normalize team names (they're already normalized in API CSV, but ensure consistency)
        filtered['home_team_normalized'] = filtered['home_team_name'].apply(normalize_team_name)
        filtered['away_team_normalized'] = filtered['away_team_name'].apply(normalize_team_name)
        
        # Create fixtures DataFrame with same structure as schedule format
        fixtures = pd.DataFrame({
            'date': filtered['date'],
            'home_team': filtered['home_team_normalized'],
            'away_team': filtered['away_team_normalized'],
            'round': filtered['matchday'].apply(lambda x: f"Matchweek {int(x)}" if pd.notna(x) else None)
        })
        
        # Deduplicate matches
        fixtures = fixtures.sort_values('date').drop_duplicates(
            subset=['home_team', 'away_team', 'date'], 
            keep='first'
        )
        
        return fixtures
    
    # MLS: Use schedule data (unchanged)
    if league_lower == 'mls':
        return pd.DataFrame()
    
    # Fallback: Use schedule for any other leagues
    schedule = load_schedule(league, 'data')
    if len(schedule) == 0:
        return pd.DataFrame()
    
    # Handle regular leagues with schedule (legacy support)
    import re
    
    df = schedule.copy()
    
    # Extract matchweek from round column
    def extract_matchweek(round_val):
        if pd.isna(round_val):
            return None
        round_str = str(round_val).strip()
        
        # Try to extract number from "Matchweek X", "Round X", "GW X", etc.
        patterns = [
            r'(?:matchweek|match\s*week|round|gw|gameweek)\s*(\d+)',
            r'(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, round_str, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        return None
    
    df['_matchweek'] = df['round'].apply(extract_matchweek) if 'round' in df.columns else None
    
    # Filter by requested matchweek
    if df['_matchweek'] is not None:
        fixtures = df[df['_matchweek'] == matchweek].copy()
    else:
        fixtures = pd.DataFrame()
    
    # Deduplicate matches
    if len(fixtures) > 0:
        fixtures = fixtures.sort_values('date').drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first')
        # Clean temp columns
        if '_matchweek' in fixtures.columns:
            fixtures = fixtures.drop('_matchweek', axis=1)
    
    return fixtures


def get_mls_fixtures_for_matchweek(schedule, matchweek):
    """Get MLS fixtures for a specific matchweek.
    
    MLS uses "Regular Season" and playoff rounds (Round One, Wild Card Round).
    We'll group ALL matches evenly into matchweeks (approximately 9-10 matches per week).
    """
    import pandas as pd
    import numpy as np
    
    # Include ALL matches (Regular Season + Playoffs)
    # Don't filter - include all round types
    all_matches = schedule.copy()
    
    if len(all_matches) == 0:
        return pd.DataFrame()
    
    # Ensure dates are datetime
    all_matches['date'] = pd.to_datetime(all_matches['date'], errors='coerce')
    all_matches = all_matches[all_matches['date'].notna()].copy()
    
    if len(all_matches) == 0:
        return pd.DataFrame()
    
    # Sort by date
    all_matches = all_matches.sort_values('date').reset_index(drop=True)
    
    # Deduplicate matches before grouping
    # This ensures team name normalization removes duplicates
    seen_matches = set()
    deduplicated_fixtures = []
    
    for _, row in all_matches.iterrows():
        match_key = (tuple(sorted([row['home_team'], row['away_team']])), row['date'])
        if match_key not in seen_matches:
            seen_matches.add(match_key)
            deduplicated_fixtures.append(row)
    
    if len(deduplicated_fixtures) == 0:
        return pd.DataFrame()
    
    all_matches = pd.DataFrame(deduplicated_fixtures).reset_index(drop=True)
    
    # Group matches evenly into matchweeks
    # This creates more consistent matchweek sizes across all weeks
    total_matches = len(all_matches)
    
    # Calculate base matches per week and remainder
    matches_per_week = total_matches // 34
    remainder = total_matches % 34
    
    # Assign matchweek numbers evenly, distributing remainder to first few weeks
    all_matches['_matchweek'] = 0
    match_idx = 0
    
    for week in range(1, 35):
        # First 'remainder' weeks get one extra match
        week_size = matches_per_week + (1 if week <= remainder else 0)
        if match_idx < len(all_matches):
            end_idx = min(match_idx + week_size, len(all_matches))
            all_matches.loc[match_idx:end_idx - 1, '_matchweek'] = week
            match_idx = end_idx
        else:
            break
    
    # Filter by requested matchweek
    fixtures = all_matches[all_matches['_matchweek'] == matchweek].copy()
    
    # Drop temporary columns
    if '_matchweek' in fixtures.columns:
        fixtures = fixtures.drop('_matchweek', axis=1)
    
    return fixtures


def get_ucl_fixtures_for_matchweek(schedule, matchweek):
    """Get UCL fixtures for a specific matchweek.
    
    PROBLEM: UCL schedule has all matches labeled as 'League phase' (no matchweek numbers).
    SOLUTION: Distribute matches evenly across 8 matchweeks based on chronological order.
    Each matchweek should have approximately 18-19 matches (152 total / 8 matchweeks).
    """
    import pandas as pd
    import re
    
    df = schedule.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # First, try to extract matchweek from 'round' column if it contains numbers
    def extract_mw(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        s = s.replace('match week', 'matchweek').replace('match day', 'matchday')
        s = re.sub(r'\s+', ' ', s)
        patterns = [
            r'(?:match\s*week|matchweek)\s*(\d+)',
            r'(?:match\s*day|matchday)\s*(\d+)',
            r'\bmd\s*(\d+)\b',
            r'\bmw\s*(\d+)\b'
        ]
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                try:
                    return int(m.group(1))
                except:
                    pass
        return None
    
    df['round'] = df['round'].astype(str)
    df['_mw'] = df['round'].apply(extract_mw)
    
    # If no matchweeks extracted (all are "League phase"), distribute evenly by date
    if df['_mw'].notna().sum() == 0:
        # Sort by date and distribute evenly across 8 matchweeks
        df = df.sort_values('date').reset_index(drop=True)
        
        # Deduplicate matches first (same teams, same date)
        df = df.drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first').reset_index(drop=True)
        
        total_matches = len(df)
        matches_per_week = total_matches // 8
        remainder = total_matches % 8
        
        # Assign matchweeks evenly, distributing remainder to first few weeks
        df['_mw'] = 0
        match_idx = 0
        
        for week in range(1, 9):  # 8 matchweeks
            # First 'remainder' weeks get one extra match
            week_size = matches_per_week + (1 if week <= remainder else 0)
            if match_idx < len(df):
                end_idx = min(match_idx + week_size, len(df))
                df.loc[match_idx:end_idx - 1, '_mw'] = week
                match_idx = end_idx
            else:
                break
    
    # Filter desired matchweek
    fixtures = df[df['_mw'] == matchweek].copy()
    
    # Deduplicate again (in case of any remaining duplicates)
    if len(fixtures) > 0:
        fixtures = fixtures.sort_values('date').drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first')
        # Clean temp columns
        if '_mw' in fixtures.columns:
            fixtures = fixtures.drop('_mw', axis=1)
    
    return fixtures


def get_max_matchweeks(league):
    """Get maximum number of matchweeks for a league."""
    max_matchweeks = {
        'epl': 38,
        'laliga': 38,
        'seriea': 38,
        'bundesliga': 34,
        'ligue1': 34,
        'mls': 34,  # MLS typically has 34 regular season games
        'ucl': 8    # UCL league phase has 8 matchdays
    }
    
    league_lower = league.lower()
    return max_matchweeks.get(league_lower, 38)  # Default to 38 if not found


@st.cache_data
def get_available_matchweeks(league):
    """Get all available matchweeks for a league.
    For top 5 leagues + UCL, uses actual matchweeks from API data.
    For others, uses max matchweeks.
    """
    league_lower = league.lower()
    api_leagues = ['epl', 'laliga', 'seriea', 'bundesliga', 'ligue1', 'ucl']
    
    if league_lower in api_leagues:
        # Load API data and get actual matchweeks
        api_df = load_api_data()
        if len(api_df) == 0:
            max_weeks = get_max_matchweeks(league)
            return list(range(1, max_weeks + 1))
        
        # Map league code
        league_code_mapping = {
            'epl': 'EPL',
            'laliga': 'LALIGA',
            'seriea': 'SERIEA',
            'bundesliga': 'BUNDESLIGA',
            'ligue1': 'LIGUE1',
            'ucl': 'UCL'
        }
        
        league_code = league_code_mapping.get(league_lower)
        if league_code:
            league_data = api_df[api_df['league_code'] == league_code]
            if len(league_data) > 0:
                available_matchweeks = sorted(league_data['matchday'].dropna().unique())
                # Convert to int and filter valid values
                available_matchweeks = [int(mw) for mw in available_matchweeks if pd.notna(mw) and mw != '']
                if available_matchweeks:
                    return available_matchweeks
    
    # Fallback: Use max matchweeks
    max_weeks = get_max_matchweeks(league)
    return list(range(1, max_weeks + 1))


def display_prediction(prediction, league_name, date_str):
    """Display prediction results in a formatted card."""
    
    st.markdown("---")
    
    # Check for new teams and display warnings
    home_strengths = prediction.get('home_strengths', {})
    away_strengths = prediction.get('away_strengths', {})
    
    home_is_new = home_strengths.get('is_new_team', False)
    away_is_new = away_strengths.get('is_new_team', False)
    
    home_name = prediction['home_team']
    away_name = prediction['away_team']
    
    if home_is_new:
        home_name = f"{prediction['home_team']} ⚠️ Newly Promoted"
    if away_is_new:
        away_name = f"{prediction['away_team']} ⚠️ Newly Promoted"
    
    st.markdown(f"## 🏟️ {home_name} vs {away_name}")
    st.markdown(f"**📅 {date_str} | {league_name}**")
    
    # Display warnings if teams are new
    if home_is_new or away_is_new:
        st.warning("⚠️ **Note:** One or more teams are newly promoted or have limited data. Predictions use league-based priors.")
    
    # Recent Form
    col1, col2 = st.columns(2)
    
    with col1:
        home_form_str = " ".join(prediction['home_form']) if prediction['home_form'] else "No recent matches"
        st.markdown(f"**{prediction['home_team']} Form:** {home_form_str}")
    
    with col2:
        away_form_str = " ".join(prediction['away_form']) if prediction['away_form'] else "No recent matches"
        st.markdown(f"**{prediction['away_team']} Form:** {away_form_str}")
    
    st.markdown("---")
    
    # Win Probabilities
    st.markdown("### 🎲 Win Probabilities")
    
    probs = prediction['probabilities']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏠 Home Win", f"{probs['home_win']}%")
        # Progress bar
        st.progress(probs['home_win'] / 100)
    
    with col2:
        st.metric("🤝 Draw", f"{probs['draw']}%")
        st.progress(probs['draw'] / 100)
    
    with col3:
        st.metric("🚗 Away Win", f"{probs['away_win']}%")
        st.progress(probs['away_win'] / 100)
    
    st.markdown("---")
    
    # Goals & Props
    st.markdown("### ⚽ Goals & Props")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("BTTS Yes", f"{probs['btts']}%")
        st.progress(probs['btts'] / 100)
    
    with col2:
        st.metric("Over 1.5 Goals", f"{probs['over_15']}%")
        st.progress(probs['over_15'] / 100)
    
    with col3:
        st.metric("Over 2.5 Goals", f"{probs['over_25']}%")
        st.progress(probs['over_25'] / 100)
    
    with col4:
        over_35 = probs.get('over_35', 0)
        st.metric("Over 3.5 Goals", f"{over_35}%")
        st.progress(over_35 / 100)
    
    # Correct Score Probabilities
    if 'correct_scores' in prediction and prediction['correct_scores']:
        st.markdown("---")
        st.markdown("### 🎯 Most Likely Correct Scores")
        correct_scores = prediction['correct_scores']
        
        cols = st.columns(min(len(correct_scores), 5))
        for idx, (score, prob) in enumerate(list(correct_scores.items())[:5]):
            with cols[idx]:
                st.metric(f"{score}", f"{prob*100:.1f}%")
    
    st.markdown("---")
    
    # Expected Stats
    if prediction['expected_stats']:
        st.markdown("### 📊 Expected Stats")
        
        stats = prediction['expected_stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{prediction['home_team']}**")
            st.write(f"Shots/90: {stats['home_shots']}")
            st.write(f"Possession%: {stats['home_poss']}%")
            st.write(f"PSxG: {stats['home_psxg']}")
            st.write(f"Crosses Stopped%: {stats['home_crosses_stp']}%")
        
        with col2:
            st.markdown(f"**{prediction['away_team']}**")
            st.write(f"Shots/90: {stats['away_shots']}")
            st.write(f"Possession%: {stats['away_poss']}%")
            st.write(f"PSxG: {stats['away_psxg']}")
            st.write(f"Crosses Stopped%: {stats['away_crosses_stp']}%")
        
        st.markdown("---")
    
    # Best Bet
    st.markdown("### 🎯 Best Bet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction['best_bet'] != "No strong bet" and prediction['best_bet'] != "No Bet":
            st.success(f"➡️ **{prediction['best_bet']}** (Value)")
        else:
            st.info("➡️ **No strong bet** (No clear value)")
    
    with col2:
        if prediction['best_prop'] != "No Value Bet":
            st.success(f"**Prop:** {prediction['best_prop']}")
        else:
            st.info("**Prop:** No Value Bet")
    
    # Team Strengths
    if 'home_strengths' in prediction and 'away_strengths' in prediction:
        st.markdown("---")
        st.markdown("### 💪 Team Strengths")
        
        home_strengths = prediction['home_strengths']
        away_strengths = prediction['away_strengths']
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_display_name = prediction['home_team']
            if home_strengths.get('is_new_team', False):
                home_display_name += " ⚠️"
            st.markdown(f"**{home_display_name}**")
            st.write(f"🏠 Home Attack: {home_strengths.get('home_attack', 'N/A')}")
            st.write(f"🚗 Away Attack: {home_strengths.get('away_attack', 'N/A')}")
            st.write(f"🛡️ Home Defense: {home_strengths.get('home_defense', 'N/A')}")
            st.write(f"🛡️ Away Defense: {home_strengths.get('away_defense', 'N/A')}")
            st.write(f"⭐ Overall Strength: {home_strengths.get('strength', 'N/A')}")
        
        with col2:
            away_display_name = prediction['away_team']
            if away_strengths.get('is_new_team', False):
                away_display_name += " ⚠️"
            st.markdown(f"**{away_display_name}**")
            st.write(f"🏠 Home Attack: {away_strengths.get('home_attack', 'N/A')}")
            st.write(f"🚗 Away Attack: {away_strengths.get('away_attack', 'N/A')}")
            st.write(f"🛡️ Home Defense: {away_strengths.get('home_defense', 'N/A')}")
            st.write(f"🛡️ Away Defense: {away_strengths.get('away_defense', 'N/A')}")
            st.write(f"⭐ Overall Strength: {away_strengths.get('strength', 'N/A')}")
    
    # Expected Goals
    st.markdown("---")
    st.markdown("### 📈 Expected Goals (λ)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{prediction['home_team']}:** {prediction['lambda_home']}")
    with col2:
        st.write(f"**{prediction['away_team']}:** {prediction['lambda_away']}")


# Sidebar
st.sidebar.header("⚙️ Settings")

league_mapping = get_league_mapping()

# League selection
league_options = {k: league_mapping.get(k, k) for k in ['EPL', 'LALIGA', 'BUNDESLIGA', 'SERIEA', 'LIGUE1', 'MLS', 'UCL']}
selected_league_display = st.sidebar.selectbox(
    "Select League",
    options=list(league_options.values()),
    index=0
)

# Get league code
selected_league = [k for k, v in league_options.items() if v == selected_league_display][0]

# Handle MLS differently (round type + date) vs other leagues (matchweek-based)
if selected_league.lower() == 'mls':
    # MLS: Round type → Date selection
    st.sidebar.subheader("📅 Select Round Type")
    available_round_types = get_mls_round_types()
    
    if len(available_round_types) == 0:
        st.sidebar.warning(f"No round types available for {selected_league_display}.")
        st.stop()
    
    selected_round_type = st.sidebar.selectbox(
        "Round Type",
        options=available_round_types,
        index=0,
        key="mls_round_type_selector"
    )
    
    # Get fixtures for selected round type
    all_mls_fixtures = get_mls_fixtures_for_round_type(selected_round_type)
    
    if len(all_mls_fixtures) == 0:
        st.sidebar.warning(f"No matches available for {selected_round_type}.")
        st.stop()
    
    # Get unique dates and sort them
    unique_dates = sorted(all_mls_fixtures['date'].dropna().unique())
    date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") if isinstance(d, (pd.Timestamp, str)) else str(d) for d in unique_dates]
    
    # Date dropdown
    st.sidebar.subheader("📅 Select Date")
    selected_date_label = st.sidebar.selectbox(
        "Date",
        options=date_labels,
        index=0,
        key="mls_date_selector"
    )
    selected_date = pd.to_datetime(selected_date_label)
    
    # Filter fixtures for selected date
    fixtures = all_mls_fixtures[
        all_mls_fixtures['date'].dt.normalize() == selected_date.normalize()
    ].copy()
    
    st.header(f"📅 Fixtures - {selected_league_display}")
    
    if len(fixtures) == 0:
        st.warning(f"❌ No matches available for {selected_round_type} on {selected_date.strftime('%Y-%m-%d')}.")
        st.info("💡 Please select a different date or round type to view fixtures.")
        st.stop()
    else:
        st.subheader(f"📅 {selected_round_type} - {selected_date.strftime('%Y-%m-%d')} ({len(fixtures)} match{'es' if len(fixtures) > 1 else ''})")
    
    selected_matchweek = None  # Not used for MLS
else:
    # Other leagues: Matchweek-based selection
    st.sidebar.subheader("📅 Select Matchweek")
    available_matchweeks = get_available_matchweeks(selected_league.lower())
    if len(available_matchweeks) == 0:
        st.sidebar.warning(f"No matchweeks available for {selected_league_display}.")
        st.stop()

    selected_matchweek = st.sidebar.selectbox(
        "Matchweek",
        options=available_matchweeks,
        index=0,
        key="matchweek_selector"
    )

    # Get fixtures for selected matchweek
    fixtures = get_fixtures_for_matchweek(selected_league.lower(), selected_matchweek)

    # Display fixtures section
    st.header(f"📅 Fixtures - {selected_league_display}")
    st.subheader(f"📅 Matchweek {selected_matchweek}")

    if len(fixtures) == 0:
        st.warning(f"❌ No matches available for {selected_league_display} - Matchweek {selected_matchweek}.")
        st.info("💡 Please select a different matchweek or league to view fixtures.")
        st.stop()
    else:
        st.success(f"✅ Found {len(fixtures)} match{'es' if len(fixtures) > 1 else ''} in Matchweek {selected_matchweek}")

# Display fixtures dropdown (for both UCL and other leagues)
if len(fixtures) > 0:
    # Create fixture options
    fixture_options = []
    for idx, row in fixtures.iterrows():
        fixture_str = f"{row['home_team']} vs {row['away_team']}"
        # For other leagues (not MLS), add round info if available
        if selected_league.lower() != 'mls' and pd.notna(row.get('round')):
            fixture_str += f" - {row['round']}"
        fixture_options.append((idx, fixture_str))
    
    # Display fixtures as selectbox
    selected_fixture_idx = st.selectbox(
        "Select Match",
        options=range(len(fixture_options)),
        format_func=lambda x: fixture_options[x][1],
        index=0,
        key="fixture_selector"
    )
    
    if selected_fixture_idx is not None:
        selected_fixture = fixtures.iloc[selected_fixture_idx]
        
        # Auto-generate prediction when fixture is selected
        # Use session state to avoid regenerating on every rerun
        # Use matchweek and teams for unique key (for MLS, use date instead of matchweek)
        if selected_league.lower() == 'mls':
            fixture_date = selected_fixture.get('date')
            date_key = pd.to_datetime(fixture_date).strftime("%Y%m%d") if pd.notna(fixture_date) else "nodate"
            prediction_key = f"pred_{selected_league}_{selected_fixture['home_team']}_{selected_fixture['away_team']}_{date_key}"
        else:
            prediction_key = f"pred_{selected_league}_{selected_fixture['home_team']}_{selected_fixture['away_team']}_mw{selected_matchweek}"
        
        if prediction_key not in st.session_state or st.session_state.get('force_recalculate', False):
            with st.spinner("Computing prediction..."):
                # Normalize team names from fixture
                home_team = normalize_team_name(selected_fixture['home_team'])
                away_team = normalize_team_name(selected_fixture['away_team'])
                
                prediction = predict_match(
                    st.session_state.master_data,
                    home_team,
                    away_team,
                    selected_league
                )
                
                if prediction:
                    st.session_state[prediction_key] = prediction
                    st.session_state['force_recalculate'] = False
                else:
                    st.error("Could not generate prediction. Please check the team names in the data.")
                    st.session_state[prediction_key] = None
        
        # Display prediction if available
        if prediction_key in st.session_state and st.session_state[prediction_key] is not None:
            prediction = st.session_state[prediction_key]
            # Get date from fixture if available, otherwise use matchweek
            fixture_date = selected_fixture.get('date')
            if pd.notna(fixture_date):
                date_str = pd.to_datetime(fixture_date).strftime("%Y-%m-%d") if isinstance(fixture_date, (pd.Timestamp, str)) else str(fixture_date)
            else:
                date_str = f"Matchweek {selected_matchweek}"
            display_prediction(prediction, selected_league_display, date_str)


# Footer
st.markdown("---")

