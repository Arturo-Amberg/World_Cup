# FIFA World Cup 2026 Analytical Terminal

A high-fidelity analytical platform for modeling, predicting, and identifying value in the 2026 FIFA World Cup.

## 🎯 Objective
To provide a professional-grade modeling environment that combines ELO ratings, Poisson distributions, and Machine Learning to deliver actionable tournament insights and betting intelligence.

## 🚀 Core Features

### 📊 Group Stage Analysis
Real-time modeling of all 12 groups. Includes advancement probabilities, expected goals (xG) breakdowns, and host nation advantage adjustments for the USA, Mexico, and Canada.

### 💰 Value Bets & Kelly Sizing
Algorithmic detection of market inefficiencies. The terminal compares model probabilities against bookmaker lines (Over/Under, BTTS, 1X2) and suggests optimal stake sizing using the Kelly Criterion.

### 🌳 Tournament Brackets
Monte Carlo simulations (100k+ runs) to generate a consensus knockout path. Visualize the most probable routes to the final for every nation.

## 🛠 Tech Stack
- **Backend**: Python / Flask
- **Modeling**: ELO, Poisson, LightGBM / CatBoost
- **Frontend**: Vanilla JS / CSS (macOS Light aesthetic)

## 📖 Documentation
- [Product Strategy (PRODUCT.md)](PRODUCT.md) — Users, features, and brand personality.
- [Design System (DESIGN.md)](DESIGN.md) — Visual theme, colors, and component library.

## 🖥 Local Development
1. Install dependencies: `pip install -r requirements.txt`
2. Run the analytical server: `python app.py`
3. Access the terminal: `http://localhost:5007`
