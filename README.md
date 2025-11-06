# ⚽ Football Prediction System

A comprehensive football betting prediction system using **Poisson + Dixon-Coles + Monte Carlo simulation** models, built with Python and Streamlit.

## 🎯 Features

- **Multi-League Support**: EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, UEFA Champions League
- **Advanced Prediction Models**:
  - Poisson distribution for goal prediction
  - Dixon-Coles adjustment for low-score correlation
  - Monte Carlo simulation (10,000 iterations)
  - Blended predictions (70% Dixon-Coles, 30% Monte Carlo)
- **Season Weighting**: Recent seasons weighted higher (exponential decay)
- **Head-to-Head Analysis**: H2H history with recency weighting
- **Expected Statistics**: Shots/90, Possession%, PSxG, Crosses Stopped%
- **Recent Form Tracking**: Visual form indicators (✅ Win, ❌ Loss, ➖ Draw)
- **Best Bet Recommendations**: Smart value betting suggestions

## 📊 Prediction Outputs

- **Win Probabilities**: Home Win / Draw / Away Win
- **Goal Markets**: BTTS, Over 1.5, Over 2.5 goals
- **Expected Stats**: Team-level match statistics
- **Recent Form**: Last 5 matches per team
- **Best Bet**: Value betting recommendations

## 🚀 Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
project/
│
├── data/
│   ├── epl_final.csv
│   ├── laliga_final.csv
│   ├── bundesliga_final.csv
│   ├── seriea_final.csv
│   ├── ligue1_final.csv
│   ├── mls_final.csv
│   ├── ucl_final.csv
│   └── [league]_Schedule.xlsx (schedule files)
│
├── model/
│   ├── __init__.py
│   ├── poisson_model.py       # Poisson probability calculations
│   ├── dixon_coles.py         # Dixon-Coles low-score adjustment
│   ├── monte_carlo.py         # Monte Carlo simulation
│   └── predictor.py           # Main prediction orchestrator
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # CSV/Excel loading and merging
│   ├── feature_engineering.py # Team/league averages, relative strengths
│   ├── weighting.py           # Season and match decay weights
│   └── metrics_calculator.py  # Expected stats and recent form
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Model Details

### 1. Data Loading
- Loads all 7 league CSVs into a master dataframe
- Handles schedule Excel files and converts to fixture format
- Standardizes dates and extracts season information

### 2. Feature Engineering
- **Team Averages**: Weighted seasonal averages for home/away performance
- **League Averages**: League-wide baselines for normalization
- **Relative Strengths**: Attack and defense strength ratios
- **H2H Analysis**: Head-to-head history with recency weighting

### 3. Season Weighting
- Formula: `weight = 1 / (1 + (current_season - season_year))`
- Recent seasons get higher importance
- Applied to all statistical calculations

### 4. Expected Goals (λ)
```
λ_home = attack_strength_home × defense_strength_away × league_home_xg_avg
λ_away = attack_strength_away × defense_strength_home × league_away_xg_avg
```

### 5. Dixon-Coles Adjustment
- Applies correction factor (ρ = -0.13) for low-score games (0-0, 1-0, 0-1, 1-1)
- Adjusts Poisson probabilities for under-prediction of low scores

### 6. Monte Carlo Simulation
- Runs 10,000 simulations using Poisson distributions
- Counts outcome frequencies
- Blended with Dixon-Coles results (70/30 split)

### 7. Best Bet Logic
- **Result Bet**: If any outcome > 60% → recommend that outcome
- **Prop Bet**: If BTTS > 65% → BTTS Yes; if Over 2.5 > 65% → Over 2.5

## 📝 Usage

### Streamlit App

1. Select **League** from sidebar dropdown
2. Select **Date** from available match dates
3. Choose a **Fixture** from the list
4. Click **"Generate Prediction"**
5. View comprehensive prediction card with:
   - Win probabilities
   - Goal markets
   - Expected statistics
   - Recent form
   - Best bet recommendations

### Manual Prediction

If no fixtures are scheduled, use the sidebar to enter teams manually:
- Enter Home Team name
- Enter Away Team name
- Click "Predict Match"

## 🔍 Data Requirements

The system expects CSV files with the following columns:
- `date`: Match date
- `team`: Team name
- `opponent`: Opponent name
- `Venue`: Home/Away
- `Result`: W/L/D
- `GF`, `GA`: Goals for/against
- `xG`, `xGA`: Expected goals
- `Poss`: Possession percentage
- `Standard_Sh`, `Standard_SoT`: Shots and shots on target
- `Performance_PSxG`: Post-shot expected goals
- `Crosses_Stp%`: Crosses stopped percentage
- `Season`: Season identifier

## ⚙️ Configuration

Key parameters can be adjusted in the code:
- **Monte Carlo simulations**: `n_simulations=10000` in `monte_carlo.py`
- **Blending weights**: `poisson_weight=0.7` in `monte_carlo.py`
- **Dixon-Coles rho**: `rho=-0.13` in `dixon_coles.py`
- **Best bet threshold**: `>= 0.60` in `predictor.py`
- **H2H weight**: `0.2` (20%) in `predictor.py`

## 🧪 Testing

Run the test script to verify the system:
```bash
python test_prediction.py
```

## 📊 Example Output

```
🏟️ Arsenal vs Chelsea
📅 November 10, 2025 | Premier League

Recent Form:
Arsenal: ✅ ✅ ❌ ➖ ❌
Chelsea: ❌ ✅ ➖ ✅ ✅

Win Probabilities:
🏠 Home Win: 31.6%
🤝 Draw: 29.8%
🚗 Away Win: 38.6%

Goals & Props:
BTTS Yes: 42.2%
Over 1.5 Goals: 78.5%
Over 2.5 Goals: 55.3%

Expected Stats:
Shots/90: 13.4 - 11.2
Possession%: 57.1 - 42.9
PSxG: 1.56 - 1.32

🎯 Best Bet:
➡️ No Bet (No clear value)
Prop: No Value Bet
```

## 🔮 Future Enhancements

- Live data API integration
- Historical prediction accuracy tracking
- Betting odds comparison
- Machine learning model integration
- Custom weighting adjustments
- Multiple model ensemble

## 📄 License

This project is built for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

---

**Built with:** Python, Streamlit, Pandas, NumPy, SciPy

