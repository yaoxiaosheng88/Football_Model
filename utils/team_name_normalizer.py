"""
Team name normalization module.
Standardizes team names across all leagues to ensure consistency.
"""

import re
import os
import json
import pandas as pd

# Path to API-derived team ID -> name mapping
TEAM_MAP_PATH = "data/team_name_map.json"


# Team name mappings for standardization
TEAM_NAME_MAPPING = {
    # EPL teams
    'west ham': 'West Ham United',
    'west ham united': 'West Ham United',
    'west ham u': 'West Ham United',
    'manchester united': 'Manchester United',
    'manchester utd': 'Manchester United',
    'manchester u': 'Manchester United',
    'manchester city': 'Manchester City',
    'man city': 'Manchester City',
    'eng manchester utd': 'Manchester United',  # Handle prefix variant
    'eng manchester city': 'Manchester City',  # Handle prefix variant
    'tottenham': 'Tottenham Hotspur',
    'tottenham hotspur': 'Tottenham Hotspur',
    'tottenham spurs': 'Tottenham Hotspur',
    'arsenal': 'Arsenal',
    'eng arsenal': 'Arsenal',  # UCL variant
    'chelsea': 'Chelsea',
    'eng chelsea': 'Chelsea',  # UCL variant
    'liverpool': 'Liverpool',
    'eng liverpool': 'Liverpool',  # UCL variant
    'brighton': 'Brighton & Hove Albion',
    'brighton and hove albion': 'Brighton & Hove Albion',
    'brighton & hove': 'Brighton & Hove Albion',
    'brighton & hove albion': 'Brighton & Hove Albion',
    'crystal palace': 'Crystal Palace',
    'newcastle': 'Newcastle United',
    'newcastle united': 'Newcastle United',
    'newcastle utd': 'Newcastle United',
    'wolves': 'Wolverhampton Wanderers',
    'wolverhampton': 'Wolverhampton Wanderers',
    'wolverhampton wanderers': 'Wolverhampton Wanderers',
    'leicester': 'Leicester City',
    'leicester city': 'Leicester City',
    'burnley': 'Burnley',
    'fulham': 'Fulham',
    'brentford': 'Brentford',
    'everton': 'Everton',
    'southampton': 'Southampton',
    'watford': 'Watford',
    'norwich': 'Norwich City',
    'norwich city': 'Norwich City',
    'leeds': 'Leeds United',
    'leeds united': 'Leeds United',
    'leeds utd': 'Leeds United',
    'aston villa': 'Aston Villa',
    'bournemouth': 'Bournemouth',
    'sunderland': 'Sunderland',
    'sunderland afc': 'Sunderland',
    'sheffield united': 'Sheffield United',
    'sheffield utd': 'Sheffield United',
    'west bromwich': 'West Bromwich Albion',
    'west brom': 'West Bromwich Albion',
    'west bromwich albion': 'West Bromwich Albion',
    'nottingham forest': 'Nottingham Forest',
    'nott\'ham forest': 'Nottingham Forest',
    'nott\'ham': 'Nottingham Forest',
    
    # La Liga teams
    'real madrid': 'Real Madrid',
    'real madrid (esp)': 'Real Madrid',
    'barcelona': 'Barcelona',
    'barcelona (esp)': 'Barcelona',
    'esp barcelona': 'Barcelona',  # UCL variant
    'atletico madrid': 'Atletico Madrid',
    'atletico': 'Atletico Madrid',
    'atlético madrid': 'Atletico Madrid',
    'club atlético de madrid': 'Atletico Madrid',
    'club atletico de madrid': 'Atletico Madrid',
    'atlético de madrid': 'Atletico Madrid',
    'atletico de madrid': 'Atletico Madrid',
    'sevilla': 'Sevilla',
    'sevilla (esp)': 'Sevilla',
    'valencia': 'Valencia',
    'valencia (esp)': 'Valencia',
    'villarreal': 'Villarreal',
    'villarreal (esp)': 'Villarreal',
    'real sociedad': 'Real Sociedad',
    'real sociedad (esp)': 'Real Sociedad',
    'real sociedad de fútbol': 'Real Sociedad',
    'real sociedad de futbol': 'Real Sociedad',
    'athletic bilbao': 'Athletic Bilbao',
    'athletic': 'Athletic Bilbao',
    'athletic club': 'Athletic Bilbao',
    'real betis': 'Real Betis',
    'real betis (esp)': 'Real Betis',
    'betis': 'Real Betis',
    'real betis balompié': 'Real Betis',
    'real betis balompie': 'Real Betis',
    'celta vigo': 'Celta Vigo',
    'rc celta de vigo': 'Celta Vigo',
    'rayo vallecano': 'Rayo Vallecano',
    'rayo vallecano de madrid': 'Rayo Vallecano',
    'getafe': 'Getafe',
    'girona': 'Girona',
    'osasuna': 'Osasuna',
    'ca osasuna': 'Osasuna',
    'mallorca': 'Mallorca',
    'rcd mallorca': 'Mallorca',
    'espanyol': 'Espanyol',
    'espanyol de barcelona': 'Espanyol',
    'cadiz': 'Cadiz',
    'alaves': 'Deportivo Alaves',
    'alavés': 'Deportivo Alaves',
    'deportivo alaves': 'Deportivo Alaves',
    'deportivo alavés': 'Deportivo Alaves',
    'real oviedo': 'Real Oviedo',
    'valencia cf': 'Valencia',
    'levante ud': 'Levante',
    'valladolid': 'Real Valladolid',
    'real valladolid': 'Real Valladolid',
    'granada': 'Granada',
    'elche': 'Elche',
    'levante': 'Levante',
    
    # Bundesliga teams
    'bayern munich': 'Bayern Munich',
    'fc bayern munich': 'Bayern Munich',
    'bayern münchen': 'Bayern Munich',
    'bayern (ger)': 'Bayern Munich',
    'de bayern munich': 'Bayern Munich',  # UCL variant
    'de bayern': 'Bayern Munich',  # UCL variant
    'borussia dortmund': 'Borussia Dortmund',
    'dortmund': 'Borussia Dortmund',
    'de dortmund': 'Borussia Dortmund',  # UCL variant
    'bvb': 'Borussia Dortmund',
    'borussia mönchengladbach': 'Borussia Monchengladbach',
    'borussia monchengladbach': 'Borussia Monchengladbach',
    'borussia mgladbach': 'Borussia Monchengladbach',
    'gladbach': 'Borussia Monchengladbach',
    'rb leipzig': 'RB Leipzig',
    'leipzig': 'RB Leipzig',
    'de rb leipzig': 'RB Leipzig',  # UCL variant
    'red bull leipzig': 'RB Leipzig',
    'bayer leverkusen': 'Bayer Leverkusen',
    'bayer 04 leverkusen': 'Bayer Leverkusen',
    'leverkusen': 'Bayer Leverkusen',
    'de leverkusen': 'Bayer Leverkusen',  # UCL variant
    'wolfsburg': 'Wolfsburg',
    'vfl wolfsburg': 'Wolfsburg',
    'eintracht frankfurt': 'Eintracht Frankfurt',
    'frankfurt': 'Eintracht Frankfurt',
    'eint frankfurt': 'Eintracht Frankfurt',
    'eintracht': 'Eintracht Frankfurt',
    'union berlin': 'Union Berlin',
    '1. fc union berlin': 'Union Berlin',
    'fc union berlin': 'Union Berlin',
    '1 fc union berlin': 'Union Berlin',
    'stuttgart': 'VfB Stuttgart',
    'vfb stuttgart': 'VfB Stuttgart',
    'freiburg': 'Freiburg',
    'hoffenheim': 'TSG Hoffenheim',
    'tsg hoffenheim': 'TSG Hoffenheim',
    'tsg 1899 hoffenheim': 'TSG Hoffenheim',
    'werder bremen': 'Werder Bremen',
    'bremen': 'Werder Bremen',
    'sv werder bremen': 'Werder Bremen',
    'augsburg': 'Augsburg',
    'mainz': 'Mainz',
    '1. fsv mainz 05': 'Mainz',
    'fsv mainz': 'Mainz',
    'mainz 05': 'Mainz',
    'bochum': 'Bochum',
    'koln': 'FC Koln',
    'köln': 'FC Koln',
    'fc koln': 'FC Koln',
    '1. fc köln': 'FC Koln',
    'hertha berlin': 'Hertha Berlin',
    'schalke': 'Schalke',
    'schalke 04': 'Schalke',
    'heidenheim': '1. FC Heidenheim',
    '1. fc heidenheim': '1. FC Heidenheim',
    '1. fc heidenheim 1846': '1. FC Heidenheim',
    'fc heidenheim': '1. FC Heidenheim',
    'st. pauli': 'FC St. Pauli',
    'st. pauli 1910': 'FC St. Pauli',
    'fc st. pauli': 'FC St. Pauli',
    'hamburger sv': 'Hamburger SV',
    'hamburger': 'Hamburger SV',
    
    # Serie A teams
    'inter milan': 'Inter Milan',
    'inter': 'Inter Milan',
    'inter (ita)': 'Inter Milan',
    'internazionale': 'Inter Milan',
    'internazionale milano': 'Inter Milan',
    'fc internazionale milano': 'Inter Milan',
    'inter milan (ita)': 'Inter Milan',
    'ac milan': 'AC Milan',
    'milan': 'AC Milan',
    'milan (ita)': 'AC Milan',
    'juventus': 'Juventus',
    'juventus (ita)': 'Juventus',
    'juventus fc': 'Juventus',
    'napoli': 'Napoli',
    'napoli (ita)': 'Napoli',
    'atalanta': 'Atalanta',
    'atalanta (ita)': 'Atalanta',
    'atalanta bc': 'Atalanta',
    'atalanta bergamo': 'Atalanta',
    'roma': 'AS Roma',
    'as roma': 'AS Roma',
    'roma (ita)': 'AS Roma',
    'lazio': 'Lazio',
    'lazio (ita)': 'Lazio',
    'fiorentina': 'Fiorentina',
    'fiorentina (ita)': 'Fiorentina',
    'bologna': 'Bologna',
    'bologna fc 1909': 'Bologna',
    'bologna fc': 'Bologna',
    'torino': 'Torino',
    'torino fc': 'Torino',
    'sassuolo': 'Sassuolo',
    'us sassuolo calcio': 'Sassuolo',
    'us sassuolo': 'Sassuolo',
    'udinese': 'Udinese',
    'udinese calcio': 'Udinese',
    'empoli': 'Empoli',
    'lecce': 'Lecce',
    'us lecce': 'Lecce',
    'salernitana': 'Salernitana',
    'monza': 'Monza',
    'verona': 'Hellas Verona',
    'hellas verona': 'Hellas Verona',
    'hellas verona fc': 'Hellas Verona',
    'sampdoria': 'Sampdoria',
    'spezia': 'Spezia',
    'genoa': 'Genoa',
    'genoa cfc': 'Genoa',
    'genoa c.f.c.': 'Genoa',
    'cagliari': 'Cagliari',
    'cagliari calcio': 'Cagliari',
    'parma': 'Parma',
    'parma calcio 1913': 'Parma',
    'parma calcio': 'Parma',
    'cremonese': 'Cremonese',
    'us cremonese': 'Cremonese',
    'como': 'Como',
    'como 1907': 'Como',
    'pisa': 'Pisa',
    'ac pisa 1909': 'Pisa',
    'pisa 1909': 'Pisa',
    
    # Ligue 1 teams
    'psg': 'Paris Saint-Germain',
    'paris saint-germain': 'Paris Saint-Germain',
    'paris sg': 'Paris Saint-Germain',
    'paris s-g': 'Paris Saint-Germain',
    'psg (fra)': 'Paris Saint-Germain',
    'paris fc': 'Paris FC',
    # Note: 'paris' alone is ambiguous - could be PSG or Paris FC
    # Only map explicit PSG variants to Paris Saint-Germain
    'monaco': 'AS Monaco',
    'as monaco': 'AS Monaco',
    'monaco (fra)': 'AS Monaco',
    'lyon': 'Lyon',
    'lyon (fra)': 'Lyon',
    'olympique lyonnais': 'Lyon',
    'marseille': 'Marseille',
    'olympique marseille': 'Marseille',
    'olympique de marseille': 'Marseille',
    'marseille (fra)': 'Marseille',
    'lille': 'Lille',
    'lille osc': 'Lille',
    'lille (fra)': 'Lille',
    'rennes': 'Rennes',
    'stade rennais': 'Rennes',
    'stade rennais fc 1901': 'Rennes',
    'stade rennais fc': 'Rennes',
    'rennes (fra)': 'Rennes',
    'nice': 'Nice',
    'ogc nice': 'Nice',
    'nice (fra)': 'Nice',
    'lens': 'Lens',
    'rc lens': 'Lens',
    'racing club de lens': 'Lens',
    'racing club lens': 'Lens',
    'lens (fra)': 'Lens',
    'monpellier': 'Montpellier',
    'montpellier': 'Montpellier',
    'monpellier (fra)': 'Montpellier',
    'toulouse': 'Toulouse FC',
    'toulouse fc': 'Toulouse FC',
    'strasbourg': 'Strasbourg',
    'rc strasbourg alsace': 'Strasbourg',
    'nantes': 'FC Nantes',
    'fc nantes': 'FC Nantes',
    'reims': 'Reims',
    'lorient': 'FC Lorient',
    'fc lorient': 'FC Lorient',
    'clermont': 'Clermont Foot',
    'clermont foot': 'Clermont Foot',
    'troyes': 'Troyes',
    'auxerre': 'Aj Auxerre',
    'aj auxerre': 'Aj Auxerre',
    'le havre': 'Le Havre AC',
    'le havre ac': 'Le Havre AC',
    'havre': 'Le Havre AC',
    'brest': 'Brest',
    'stade brestois 29': 'Brest',
    'metz': 'FC Metz',
    'fc metz': 'FC Metz',
    'angers': 'Angers SCO',
    'angers sco': 'Angers SCO',
    'angers sc': 'Angers SCO',
    
    # MLS teams
    'atlanta united': 'Atlanta United FC',
    'atlanta united fc': 'Atlanta United FC',
    'atlanta': 'Atlanta United FC',
    'atlanta utd': 'Atlanta United FC',
    'atlanta utd fc': 'Atlanta United FC',
    'cf montreal': 'CF Montreal',
    'cf montréal': 'CF Montreal',
    'montreal': 'CF Montreal',
    'montreal impact': 'CF Montreal',
    'charlotte': 'Charlotte FC',
    'charlotte fc': 'Charlotte FC',
    'charlotte football club': 'Charlotte FC',
    'chicago fire': 'Chicago Fire FC',
    'chicago fire fc': 'Chicago Fire FC',
    'chicago': 'Chicago Fire FC',
    'fc cincinnati': 'FC Cincinnati',
    'cincinnati': 'FC Cincinnati',
    'colorado rapids': 'Colorado Rapids',
    'rapids': 'Colorado Rapids',
    'columbus crew': 'Columbus Crew',
    'columbus crew sc': 'Columbus Crew',
    'crew': 'Columbus Crew',
    'crew sc': 'Columbus Crew',
    'dc united': 'DC United',
    'd.c. united': 'DC United',
    'dc': 'DC United',
    'dc united fc': 'DC United',
    'fc dallas': 'FC Dallas',
    'dallas': 'FC Dallas',
    'houston dynamo': 'Houston Dynamo FC',
    'houston dynamo fc': 'Houston Dynamo FC',
    'houston': 'Houston Dynamo FC',
    'dynamo': 'Houston Dynamo FC',
    'inter miami': 'Inter Miami CF',
    'inter miami cf': 'Inter Miami CF',
    'miami': 'Inter Miami CF',
    'miami cf': 'Inter Miami CF',
    'miami inter': 'Inter Miami CF',
    'sporting kansas city': 'Sporting Kansas City',
    'sporting kc': 'Sporting Kansas City',
    'kansas city': 'Sporting Kansas City',
    'sporting kansas': 'Sporting Kansas City',
    'skc': 'Sporting Kansas City',
    'la galaxy': 'LA Galaxy',
    'los angeles galaxy': 'LA Galaxy',
    'galaxy': 'LA Galaxy',
    'lafc': 'Los Angeles FC',
    'los angeles fc': 'Los Angeles FC',
    'la fc': 'Los Angeles FC',
    'los angeles': 'Los Angeles FC',
    'minnesota united': 'Minnesota United FC',
    'minnesota united fc': 'Minnesota United FC',
    'minnesota': 'Minnesota United FC',
    'minnesota utd': 'Minnesota United FC',
    'minnesota utd fc': 'Minnesota United FC',
    'nashville sc': 'Nashville SC',
    'nashville': 'Nashville SC',
    'nashville fc': 'Nashville SC',
    'new england revolution': 'New England Revolution',
    'ne revolution': 'New England Revolution',
    'new england': 'New England Revolution',
    'revolution': 'New England Revolution',
    'new england revs': 'New England Revolution',
    'new york city fc': 'New York City FC',
    'nycfc': 'New York City FC',
    'new york city': 'New York City FC',
    'ny city': 'New York City FC',
    'new york red bulls': 'New York Red Bulls',
    'ny red bulls': 'New York Red Bulls',
    'red bulls': 'New York Red Bulls',
    'new york rb': 'New York Red Bulls',
    'orlando city': 'Orlando City SC',
    'orlando city sc': 'Orlando City SC',
    'orlando': 'Orlando City SC',
    'orlando city fc': 'Orlando City SC',
    'philadelphia union': 'Philadelphia Union',
    'philadelphia': 'Philadelphia Union',
    'philadelphia union fc': 'Philadelphia Union',
    'union': 'Philadelphia Union',
    'portland timbers': 'Portland Timbers',
    'portland': 'Portland Timbers',
    'portland timbers fc': 'Portland Timbers',
    'timbers': 'Portland Timbers',
    'real salt lake': 'Real Salt Lake',
    'rsl': 'Real Salt Lake',
    'salt lake': 'Real Salt Lake',
    'real salt lake fc': 'Real Salt Lake',
    'san jose earthquakes': 'San Jose Earthquakes',
    'sj earthquakes': 'San Jose Earthquakes',
    'san jose': 'San Jose Earthquakes',
    'earthquakes': 'San Jose Earthquakes',
    'san jose quakes': 'San Jose Earthquakes',
    'seattle sounders': 'Seattle Sounders FC',
    'seattle sounders fc': 'Seattle Sounders FC',
    'seattle': 'Seattle Sounders FC',
    'sounders': 'Seattle Sounders FC',
    'st louis city': 'St. Louis City SC',
    'st louis city fc': 'St. Louis City SC',
    'st. louis city sc': 'St. Louis City SC',
    'st. louis city fc': 'St. Louis City SC',
    'st. louis': 'St. Louis City SC',
    'st louis': 'St. Louis City SC',
    'louis city fc': 'St. Louis City SC',  # Fix for missing "St." prefix
    'louis city': 'St. Louis City SC',
    'stl city sc': 'St. Louis City SC',
    'stl city': 'St. Louis City SC',
    'toronto fc': 'Toronto FC',
    'toronto': 'Toronto FC',
    'vancouver whitecaps': 'Vancouver Whitecaps FC',
    'vancouver whitecaps fc': 'Vancouver Whitecaps FC',
    "vancouver w'caps": 'Vancouver Whitecaps FC',
    'whitecaps': 'Vancouver Whitecaps FC',
    'vancouver': 'Vancouver Whitecaps FC',
    'austin fc': 'Austin FC',
    'austin': 'Austin FC',
    'san diego fc': 'San Diego FC',
    'san diego': 'San Diego FC',
    
    # UCL/European teams
    'ajax': 'Ajax',
    'ajax amsterdam': 'Ajax',
    'galatasaray': 'Galatasaray',
    'galatasaray sk': 'Galatasaray',
    'fenerbahce': 'Fenerbahce',
    'fenerbahçe': 'Fenerbahce',
    'shakhtar donetsk': 'Shakhtar Donetsk',
    'dinamo zagreb': 'Dinamo Zagreb',
    'dinamo kyiv': 'Dinamo Kyiv',
    'dinamo kiev': 'Dinamo Kyiv',
    'red star belgrade': 'Red Star Belgrade',
    'crvena zvezda': 'Red Star Belgrade',
    'partizan': 'Partizan',
    'partizan belgrade': 'Partizan',
    'red bull salzburg': 'Red Bull Salzburg',
    'rb salzburg': 'Red Bull Salzburg',
    'salzburg': 'Red Bull Salzburg',
    'club brugge': 'Club Brugge',
    'club brugge kv': 'Club Brugge',
    'brugge': 'Club Brugge',
    'fc copenhagen': 'Copenhagen',
    'copenhagen': 'Copenhagen',
    'fc københavn': 'Copenhagen',
    'young boys': 'Young Boys',
    'yb': 'Young Boys',
    'celtic': 'Celtic',
    'rangers': 'Rangers',
    'porto': 'FC Porto',
    'fc porto': 'FC Porto',
    'benfica': 'Benfica',
    'sport lisboa e benfica': 'Benfica',
    'sl benfica': 'Benfica',
    'sporting lisbon': 'Sporting CP',
    'sporting cp': 'Sporting CP',
    'sporting': 'Sporting CP',
    'sporting clube de portugal': 'Sporting CP',
    'sporting clube portugal': 'Sporting CP',
    'braga': 'Braga',
    'sc braga': 'Braga',
    'psv': 'PSV Eindhoven',
    'psv eindhoven': 'PSV Eindhoven',
    'nl psv eindhoven': 'PSV Eindhoven',
    'feyenoord': 'Feyenoord',
    'fc basel': 'FC Basel',
    'basel': 'FC Basel',
    'bodø/glimt': 'Bodø/Glimt',
    'bodo/glimt': 'Bodø/Glimt',
    'bodoglimt': 'Bodø/Glimt',
    'fk bodø/glimt': 'Bodø/Glimt',
    'fk bodo/glimt': 'Bodø/Glimt',
    'antwerp': 'Antwerp',
    'royal antwerp': 'Antwerp',
    'royal antwerp fc': 'Antwerp',
    'royale union saint-gilloise': 'Royale Union Saint-Gilloise',
    'union saint-gilloise': 'Royale Union Saint-Gilloise',
    'usg': 'Royale Union Saint-Gilloise',
    'paphos': 'Paphos',
    'paphos fc': 'Paphos',
    'eintracht frankfurt': 'Eintracht Frankfurt',
    'frankfurt': 'Eintracht Frankfurt',
    'atalanta': 'Atalanta',
    'atalanta bergamo': 'Atalanta',
    'napoli': 'Napoli',
    'ssc napoli': 'Napoli',
    'roma': 'Roma',
    'as roma': 'Roma',
    'lazio': 'Lazio',
    'ss lazio': 'Lazio',
    'fiorentina': 'Fiorentina',
    'acf fiorentina': 'Fiorentina',
    'bologna': 'Bologna',
    'bologna fc': 'Bologna',
    'seville': 'Sevilla',
    'sevilla fc': 'Sevilla',
    'real sociedad': 'Real Sociedad',
    'villarreal': 'Villarreal',
    'cf villarreal': 'Villarreal',
    'es villarreal': 'Villarreal',
    'villareal': 'Villarreal',
    'athletic bilbao': 'Athletic Bilbao',
    'athletic club': 'Athletic Bilbao',
    'lens': 'Lens',
    'rc lens': 'Lens',
    'marseille': 'Marseille',
    'olympique marseille': 'Marseille',
    'olympiacos': 'Olympiacos',
    'olympiakos': 'Olympiacos',
    'pae olympiakos sfp': 'Olympiacos',
    'olympiakos sfp': 'Olympiacos',
    'paok': 'PAOK',
    'paok thessaloniki': 'PAOK',
    
    # Czech teams
    'slavia prague': 'Slavia Prague',
    'slavia praha': 'Slavia Prague',
    'sk slavia praha': 'Slavia Prague',
    'sk slavia prague': 'Slavia Prague',
    'sparta prague': 'Sparta Prague',
    'sparta praha': 'Sparta Prague',
    
    # Kazakh teams
    'kairat almaty': 'Kairat Almaty',
    'qairat almaty': 'Kairat Almaty',
    'fc kairat': 'Kairat Almaty',
    'fk kairat': 'Kairat Almaty',
    'fc qairat': 'Kairat Almaty',
    'kairat': 'Kairat Almaty',
    'qairat': 'Kairat Almaty',
    'kairat army': 'Kairat Almaty',
    'qairat army': 'Kairat Almaty',
    'kairat almaty army': 'Kairat Almaty',
    'qairat almaty army': 'Kairat Almaty',
    'fc kairat army': 'Kairat Almaty',
    'fc qairat army': 'Kairat Almaty',
    'qarabağ ağdam fk': 'Qarabağ',
    'qarabağ agdam fk': 'Qarabağ',
    'qarabağ': 'Qarabağ',
    'qarabag': 'Qarabağ',
    'qarabag agdam': 'Qarabağ',
    
    # Other Eastern European teams
    'fc sheriff': 'Sheriff Tiraspol',
    'sheriff tiraspol': 'Sheriff Tiraspol',
    'sheriff': 'Sheriff Tiraspol',
    'fc astana': 'FC Astana',
    'astana': 'FC Astana',
    'fc krasnodar': 'FC Krasnodar',
    'krasnodar': 'FC Krasnodar',
    'fc rostov': 'FC Rostov',
    'rostov': 'FC Rostov',
    'fc zenit': 'Zenit Saint Petersburg',
    'zenit': 'Zenit Saint Petersburg',
    'zenit st petersburg': 'Zenit Saint Petersburg',
    'zenit saint petersburg': 'Zenit Saint Petersburg',
    'lokomotiv moscow': 'Lokomotiv Moscow',
    'cska moscow': 'CSKA Moscow',
    'spartak moscow': 'Spartak Moscow',
}


def normalize_team_name(team_name):
    """
    Normalize a team name to a standard format.
    
    Args:
        team_name: Raw team name string
        
    Returns:
        str: Normalized team name
    """
    if pd.isna(team_name) or not team_name:
        return team_name
    
    # Convert to string and strip whitespace
    name = str(team_name).strip()
    
    # Remove country prefixes (e.g., "eng Chelsea", "de Bayern Munich", "dk FC Copenhagen")
    # Pattern: 2-3 letter country code + space + team name
    # Also handle "ES" (Equipo de) prefix for Spanish teams
    # Remove country prefixes (2-3 letter codes and common variations)
    # This includes: eng, de, dk, esp, ita, fra, ger, por, ned, bel, ukr, rus, tur, gre, pol, sco, wal
    # Also handles: es (Spanish), nl (Dutch), it (Italian), fr (French), pt (Portuguese), etc.
    # Remove country prefixes - comprehensive list of country codes
    # This includes standard 2-3 letter ISO codes and common variations
    country_codes = [
        'eng', 'en', 'uk', 'de', 'dk', 'esp', 'es', 'sp', 'ita', 'it', 'fra', 'fr', 'ger', 'ge',
        'por', 'pt', 'ned', 'nl', 'bel', 'be', 'ukr', 'ru', 'rus', 'tur', 'tr', 'gre', 'gr',
        'pol', 'pl', 'sco', 'wal', 'nir', 'sui', 'ch', 'aus', 'at', 'swe', 'se', 'nor', 'no',
        'den', 'fi', 'fin', 'isl', 'is', 'cro', 'hr', 'ser', 'rs', 'hun', 'hu', 'cze', 'cz',
        'aut', 'slo', 'si', 'srb', 'rou', 'ro', 'bul', 'bg', 'mkd', 'mk', 'alb', 'al', 'bos',
        'ba', 'geo', 'ge', 'arm', 'am', 'aze', 'az', 'kaz', 'kz', 'isr', 'il', 'cyp', 'cy',
        'mlt', 'mt', 'lux', 'lu', 'lie', 'li', 'mne', 'me', 'est', 'ee', 'lat', 'lv', 'lit',
        'lt', 'ire', 'ie', 'mda', 'md', 'and', 'ad', 'smo', 'sm', 'sao', 'st', 'gib', 'gi',
        'far', 'fo', 'kos', 'xk', 'mon', 'mc', 'vat', 'va'
    ]
    # Create regex pattern - match any of these codes followed by space or hyphen
    codes_pattern = '|'.join(country_codes)
    name = re.sub(r'^(' + codes_pattern + r')[\s-]+', '', name, flags=re.IGNORECASE).strip()
    
    # Remove country prefixes in parentheses (e.g., "Real Madrid (ESP)" -> "Real Madrid")
    name = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()
    
    # Normalize spacing
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Special handling: normalize "qairat" to "kairat" before lookup
    # This ensures "qairat almaty" and "kairat almaty" are treated as the same
    name = re.sub(r'\bqairat\b', 'kairat', name, flags=re.IGNORECASE)
    
    # Create lookup key (lowercase, remove extra spaces)
    lookup_key = re.sub(r'\s+', ' ', name.lower()).strip()
    
    # Check mapping
    if lookup_key in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[lookup_key]
    
    # If not in mapping, try removing FC/CF/SC/AS/AC prefixes and checking again
    # This handles cases where one CSV has "FC Bayern Munich" and another has "Bayern Munich"
    name_no_prefix = re.sub(r'^(fc|cf|sc|as|ac)\s+', '', lookup_key, flags=re.IGNORECASE).strip()
    if name_no_prefix and name_no_prefix in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[name_no_prefix]
    
    # Try adding FC prefix if it doesn't exist and check reverse
    if not lookup_key.startswith('fc ') and not lookup_key.startswith('cf ') and not lookup_key.startswith('sc '):
        for prefix in ['fc ', 'cf ', 'sc ']:
            prefixed_version = f'{prefix}{lookup_key}'
            if prefixed_version in TEAM_NAME_MAPPING:
                return TEAM_NAME_MAPPING[prefixed_version]
    
    # If not in mapping, try to normalize common patterns
    # Remove trailing common suffixes if not in mapping
    normalized = name
    
    # Capitalize properly (first letter of each word)
    words = normalized.split()
    normalized = ' '.join(word.capitalize() if word.lower() not in ['fc', 'cf', 'sc', 'utd', 'united', 'city'] 
                          else word.upper() if word.lower() in ['fc', 'cf', 'sc'] 
                          else word.capitalize() 
                          for word in words)
    
    # Special handling for abbreviations
    if normalized.lower() == 'utd':
        normalized = 'United'
    elif normalized.lower() == 'u':
        normalized = 'United'
    
    return normalized


def normalize_dataframe_teams(df, team_column='team', opponent_column='opponent'):
    """
    Normalize team names in a dataframe.
    
    Args:
        df: DataFrame with team names
        team_column: Name of the team column
        opponent_column: Name of the opponent column
        
    Returns:
        pd.DataFrame: DataFrame with normalized team names
    """
    import pandas as pd
    
    df = df.copy()
    
    if team_column in df.columns:
        df[team_column] = df[team_column].apply(normalize_team_name)
    
    if opponent_column in df.columns:
        df[opponent_column] = df[opponent_column].apply(normalize_team_name)
    
    return df


def sync_api_team_map_with_normalizer():
    """
    Synchronize API team map (data/team_name_map.json) with the FBref-style
    normalization used here:
      - Load JSON if present
      - Re-normalize all values using normalize_team_name
      - Extend TEAM_NAME_MAPPING with any new API aliases not already covered
      - Overwrite the JSON file with normalized values (idempotent)
    """
    # Load existing API team map
    api_team_map = {}
    if os.path.exists(TEAM_MAP_PATH):
        try:
            with open(TEAM_MAP_PATH, "r", encoding="utf-8") as f:
                api_team_map = json.load(f) or {}
        except Exception:
            api_team_map = {}

    if not isinstance(api_team_map, dict):
        # Safety: if file content isn't a dict, do nothing
        print("[WARNING] team_name_map.json content is not a dict; skipping sync.")
        return

    # Re-normalize values and collect additions for TEAM_NAME_MAPPING
    additions = 0
    for team_id, api_name in list(api_team_map.items()):
        normalized_name = normalize_team_name(api_name)
        api_team_map[team_id] = normalized_name

        # Extend alias dictionary: keys in TEAM_NAME_MAPPING are lowercase aliases
        alias_key = str(api_name).strip().lower()
        # Only add if alias not already present and not already mapped to same canonical
        if alias_key and alias_key not in TEAM_NAME_MAPPING:
            # Avoid adding if some existing alias already maps to same canonical name
            if normalized_name not in TEAM_NAME_MAPPING.values():
                TEAM_NAME_MAPPING[alias_key] = normalized_name
                additions += 1
            else:
                # If canonical already exists in values, we can still add the alias
                TEAM_NAME_MAPPING[alias_key] = normalized_name
                additions += 1

    # Save back the normalized API map (idempotent)
    try:
        os.makedirs(os.path.dirname(TEAM_MAP_PATH), exist_ok=True)
        with open(TEAM_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(api_team_map, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[WARNING] Failed to write normalized team_name_map.json: {e}")
        return

    print("[OK] Re-normalized team_name_map.json with FBref-standard names via team_name_normalizer.")
    print(f"[OK] Added {additions} new aliases to normalization dictionary.")


if __name__ == "__main__":
    # Allow running this module directly to sync JSON and mapping
    sync_api_team_map_with_normalizer()

