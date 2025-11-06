"""
Football-Data.org API integration for fetching match data.

Fetches fixtures and results for 6 leagues (2025 season) and stores them in a single CSV file.
"""

import requests
import pandas as pd
import time
import os
import json
import shutil
import re
from pathlib import Path
from utils.team_name_normalizer import normalize_team_name

# Base URL for Football-Data.org
BASE_URL = "https://api.football-data.org/v4"

# Get API key from environment variable
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY:
    print("[WARNING] FOOTBALL_DATA_API_KEY not set. API calls may fail.")
    print("Set it with: $env:FOOTBALL_DATA_API_KEY='YOUR_KEY' or in Streamlit secrets")

HEADERS = {"X-Auth-Token": API_KEY} if API_KEY else {}

# Define the leagues to fetch
LEAGUES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Ligue 1": "FL1",
    "UEFA Champions League": "CL"
}

SEASON = 2025
RATE_LIMIT_DELAY = 1.5  # seconds between requests (1-2 seconds)


def fetch_league_matches(league_code: str, season: int = 2025):
    """Fetch all matches for a specific league and season."""
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {"season": season}
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if response.status_code != 200:
            print(f"⚠️ Error fetching {league_code}: HTTP {response.status_code}")
            if response.status_code == 429:
                print(f"   Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                # Retry once
                response = requests.get(url, headers=HEADERS, params=params, timeout=15)
                if response.status_code != 200:
                    print(f"   Still failed after retry")
                    return []
            else:
                print(f"   Response: {response.text[:200]}")
                return []
        return response.json().get("matches", [])
    except Exception as e:
        print(f"❌ Exception fetching {league_code}: {e}")
        return []


def load_team_name_mapping():
    """Load team ID to normalized name mapping from JSON file."""
    mapping_path = Path("data/team_name_map.json")
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load team_name_map.json: {e}")
            return {}
    return {}


def save_team_name_mapping(team_name_map):
    """Save team ID to normalized name mapping to JSON file."""
    mapping_path = Path("data/team_name_map.json")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(team_name_map, f, indent=4, ensure_ascii=False)


def extract_match_data(matches, league_name, team_name_map):
    """Extract specified columns from API response and normalize team names."""
    rows = []
    for m in matches:
        home_team = m.get("homeTeam", {})
        away_team = m.get("awayTeam", {})
        score = m.get("score", {}).get("fullTime", {})
        
        # Get team IDs and names
        home_team_id = home_team.get("id")
        home_team_name_raw = home_team.get("name")
        away_team_id = away_team.get("id")
        away_team_name_raw = away_team.get("name")
        
        # Normalize home team name using ID-based mapping
        # Always normalize to ensure consistency, even if team ID exists in map
        if home_team_id and home_team_name_raw:
            # Pre-process: remove leading/trailing prefixes/suffixes before normalizing
            name_cleaned = re.sub(r'\s+(fc|cf|sc|as|ac|afc)$', '', home_team_name_raw, flags=re.IGNORECASE).strip()
            name_cleaned = re.sub(r'^(afc|rcd|ca|ac|as|fc|cf|sc)\s+', '', name_cleaned, flags=re.IGNORECASE).strip()
            name_cleaned = re.sub(r'\s+(afc)$', '', name_cleaned, flags=re.IGNORECASE).strip()
            normalized_home = normalize_team_name(name_cleaned) if name_cleaned else None
            if normalized_home:
                team_name_map[home_team_id] = normalized_home
                home_team_name_normalized = normalized_home
            else:
                home_team_name_normalized = team_name_map.get(home_team_id, home_team_name_raw)
        else:
            home_team_name_normalized = team_name_map.get(home_team_id, home_team_name_raw)
        
        # Normalize away team name using ID-based mapping
        # Always normalize to ensure consistency, even if team ID exists in map
        if away_team_id and away_team_name_raw:
            # Pre-process: remove leading/trailing prefixes/suffixes before normalizing
            name_cleaned = re.sub(r'\s+(fc|cf|sc|as|ac|afc)$', '', away_team_name_raw, flags=re.IGNORECASE).strip()
            name_cleaned = re.sub(r'^(afc|rcd|ca|ac|as|fc|cf|sc)\s+', '', name_cleaned, flags=re.IGNORECASE).strip()
            name_cleaned = re.sub(r'\s+(afc)$', '', name_cleaned, flags=re.IGNORECASE).strip()
            normalized_away = normalize_team_name(name_cleaned) if name_cleaned else None
            if normalized_away:
                team_name_map[away_team_id] = normalized_away
                away_team_name_normalized = normalized_away
            else:
                away_team_name_normalized = team_name_map.get(away_team_id, away_team_name_raw)
        else:
            away_team_name_normalized = team_name_map.get(away_team_id, away_team_name_raw)
        
        rows.append({
            "league": league_name,
            "utcDate": m.get("utcDate"),
            "status": m.get("status"),
            "matchday": m.get("matchday"),
            "home_team_id": home_team_id,
            "home_team_name": home_team_name_normalized,
            "away_team_id": away_team_id,
            "away_team_name": away_team_name_normalized,
            "score_fullTime_home": score.get("home"),
            "score_fullTime_away": score.get("away"),
            "score_winner": m.get("score", {}).get("winner")
        })
    return rows


def save_to_csv(data, filename="data/api_results_basic.csv"):
    """Save all data to a single CSV file (overwrites existing file)."""
    if not data:
        print("⚠️ No data to save")
        return
    
    df = pd.DataFrame(data)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} matches to {filename}")


def fetch_all_leagues():
    """Fetch and save all league data into one CSV."""
    if not API_KEY:
        print("❌ API key not found! Set FOOTBALL_DATA_API_KEY environment variable")
        print("Example: $env:FOOTBALL_DATA_API_KEY='YOUR_KEY'")
        return
    
    # Load existing team name mapping
    team_name_map = load_team_name_mapping()
    initial_mapping_size = len(team_name_map)
    print(f"📋 Loaded {initial_mapping_size} existing team name mappings")
    
    print(f"🔍 Fetching 2025 season data for {len(LEAGUES)} leagues...")
    print(f"   Rate limit delay: {RATE_LIMIT_DELAY}s between requests\n")
    
    all_data = []
    
    for idx, (league_name, code) in enumerate(LEAGUES.items(), 1):
        print(f"[{idx}/{len(LEAGUES)}] Fetching {league_name} ({code})...", end=" ")
        matches = fetch_league_matches(code, SEASON)
        league_rows = extract_match_data(matches, league_name, team_name_map)
        all_data.extend(league_rows)
        print(f"✅ Found {len(league_rows)} matches")
        
        # Respect rate limit: sleep between requests (except after last league)
        if idx < len(LEAGUES):
            time.sleep(RATE_LIMIT_DELAY)
    
    # Save updated team name mapping
    new_mappings = len(team_name_map) - initial_mapping_size
    save_team_name_mapping(team_name_map)
    if new_mappings > 0:
        print(f"\n📝 Added {new_mappings} new team name mappings")
    
    # Save CSV with normalized names
    save_to_csv(all_data)
    print(f"\n✅ All {len(LEAGUES)} leagues data fetched and saved successfully to data/api_results_basic.csv")
    print(f"   Total matches: {len(all_data)}")
    print(f"\n✅ Team names normalized using team IDs.")
    print(f"   Mapping saved to data/team_name_map.json for consistent naming across future runs.")


def renormalize_team_names():
    """
    Re-normalize all team names using normalize_team_name() based on CSV data.
    Creates backups, rebuilds mapping, and updates both JSON and CSV files.
    Idempotent: running multiple times produces the same result.
    """
    # Set UTF-8 encoding for Windows console compatibility
    import sys
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass  # If reconfigure fails, continue anyway
    
    print("Re-normalizing team names using normalize_team_name()...")
    print("   Creating backups...")
    
    # Step 1: Create backups
    csv_path = "data/api_results_basic.csv"
    json_path = "data/team_name_map.json"
    
    if os.path.exists(json_path):
        shutil.copy(json_path, f"{json_path}.bak")
        print(f"   [OK] Backed up {json_path}")
    
    if os.path.exists(csv_path):
        shutil.copy(csv_path, f"{csv_path}.bak")
        print(f"   [OK] Backed up {csv_path}")
    else:
        print(f"   [WARNING] {csv_path} not found. Cannot proceed.")
        return
    
    # Step 2: Load existing data
    print("\nLoading data...")
    df = pd.read_csv(csv_path, dtype=str)
    print(f"   Loaded {len(df)} rows from CSV")
    
    # Load existing mapping if present (for comparison)
    old_team_name_map = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_team_name_map = json.load(f)
            print(f"   Loaded {len(old_team_name_map)} existing mappings from JSON")
        except Exception as e:
            print(f"   [WARNING] Could not load existing JSON: {e}")
    
    # Step 3: Rebuild mapping from CSV authoritative fields
    print("\nRebuilding team name mapping from CSV...")
    pairs = {}  # team_id -> raw_team_name (last seen)
    
    # Collect all unique (team_id, raw_team_name) pairs
    for _, row in df.iterrows():
        home_id = str(row.get("home_team_id", "")).strip()
        home_name = str(row.get("home_team_name", "")).strip()
        away_id = str(row.get("away_team_id", "")).strip()
        away_name = str(row.get("away_team_name", "")).strip()
        
        if home_id and home_id != "nan" and home_name and home_name != "nan":
            pairs[home_id] = home_name
        if away_id and away_id != "nan" and away_name and away_name != "nan":
            pairs[away_id] = away_name
    
    print(f"   Found {len(pairs)} unique team IDs")
    
    # Rebuild mapping using normalize_team_name
    team_name_map = {}
    updates = []  # Track changes for logging
    
    for team_id, raw_name in pairs.items():
        # Pre-process: remove leading and trailing common prefixes/suffixes before normalizing
        # Remove trailing: FC, CF, SC, AS, AC, AFC, etc.
        name_cleaned = re.sub(r'\s+(fc|cf|sc|as|ac|afc)$', '', raw_name, flags=re.IGNORECASE).strip()
        # Remove leading: AFC, RCD, CA, etc. (with space after)
        name_cleaned = re.sub(r'^(afc|rcd|ca|ac|as|fc|cf|sc)\s+', '', name_cleaned, flags=re.IGNORECASE).strip()
        # Remove trailing AFC (no space before, like "Sunderland Afc")
        name_cleaned = re.sub(r'\s+(afc)$', '', name_cleaned, flags=re.IGNORECASE).strip()
        normalized = normalize_team_name(name_cleaned) if name_cleaned else None
        
        if normalized:
            # Always overwrite with normalized value (don't trust old mapping)
            old_value = old_team_name_map.get(team_id)
            if old_value != normalized:
                updates.append((team_id, old_value, normalized))
            team_name_map[team_id] = normalized
    
    print(f"   Normalized {len(team_name_map)} team names")
    
    # Step 4: Apply mapping to DataFrame
    print("\nApplying normalized names to CSV...")
    
    def apply_normalized_name(row, team_id_col, team_name_col):
        """Helper to apply normalized name from mapping."""
        team_id = str(row.get(team_id_col, "")).strip()
        if team_id and team_id != "nan" and team_id in team_name_map:
            return team_name_map[team_id]
        # Fallback: normalize the raw name if ID not in mapping
        raw_name = str(row.get(team_name_col, "")).strip()
        if raw_name and raw_name != "nan":
            # Remove trailing and leading prefixes/suffixes before normalizing
            name_cleaned = re.sub(r'\s+(fc|cf|sc|as|ac|afc)$', '', raw_name, flags=re.IGNORECASE).strip()
            name_cleaned = re.sub(r'^(afc|rcd|ca|ac|as|fc|cf|sc)\s+', '', name_cleaned, flags=re.IGNORECASE).strip()
            name_cleaned = re.sub(r'\s+(afc)$', '', name_cleaned, flags=re.IGNORECASE).strip()
            return normalize_team_name(name_cleaned) if name_cleaned else raw_name
        return raw_name
    
    # CRITICAL FIX: For team ID 1045.0, explicitly set to "Paris FC" 
    # This fixes the issue where "Paris" alone doesn't normalize correctly
    if "1045.0" in team_name_map:
        team_name_map["1045.0"] = "Paris FC"
    
    df["home_team_name"] = df.apply(
        lambda row: apply_normalized_name(row, "home_team_id", "home_team_name"),
        axis=1
    )
    df["away_team_name"] = df.apply(
        lambda row: apply_normalized_name(row, "away_team_id", "away_team_name"),
        axis=1
    )
    
    # Step 5: Save outputs
    print("\nSaving updated files...")
    
    # Save JSON mapping
    json_path_obj = Path(json_path)
    json_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(team_name_map, f, indent=4, ensure_ascii=False)
    print(f"   [OK] Saved {len(team_name_map)} mappings to {json_path}")
    
    # Save CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"   [OK] Saved {len(df)} rows to {csv_path}")
    
    # Step 6: Logging & summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Unique team IDs processed: {len(pairs)}")
    print(f"Mappings in JSON: {len(team_name_map)}")
    print(f"Mappings updated: {len(updates)}")
    
    if updates:
        print(f"\nSample of updated mappings (first 10):")
        for team_id, old_val, new_val in updates[:10]:
            old_display = old_val if old_val else "(none)"
            print(f"   ID {team_id}: {old_display} -> {new_val}")
    
    print(f"\n[OK] Team name map rebuilt using normalize_team_name(). JSON and CSV updated. Backups saved as .bak.")
    print(f"{'='*60}")


def refresh_fixtures_for_streamlit():
    """
    Wrapper function for Streamlit to fetch API data and normalize team names.
    Returns a tuple: (success: bool, message: str, details: dict)
    """
    import sys
    
    # Set UTF-8 encoding for Windows console compatibility
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    try:
        # Get API key from Streamlit secrets or environment variable
        api_key = None
        
        # Try Streamlit secrets first
        try:
            import streamlit as st
            api_key = st.secrets.get("football_data", {}).get("api_key")
        except:
            pass
        
        # Fallback to environment variable
        if not api_key:
            api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        
        if not api_key:
            return False, "❌ API key not found", {
                "error": "FOOTBALL_DATA_API_KEY not set. Please configure it in Streamlit secrets or environment variables."
            }
        
        # Update global variables for this run
        global API_KEY, HEADERS
        API_KEY = api_key
        HEADERS = {"X-Auth-Token": API_KEY}
        
        # Fetch all leagues
        fetch_all_leagues()
        
        # Re-normalize team names
        renormalize_team_names()
        
        return True, "✅ Fixtures refreshed successfully", {
            "message": "API data fetched and team names normalized"
        }
        
    except Exception as e:
        error_msg = str(e)
        return False, f"❌ Error refreshing fixtures: {error_msg}", {
            "error": error_msg
        }


if __name__ == "__main__":
    import sys
    
    # Check if renormalize mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "renormalize":
        renormalize_team_names()
    else:
        fetch_all_leagues()
