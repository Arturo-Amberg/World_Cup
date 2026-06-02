#!/bin/bash
set -e

echo "==============================================="
echo " STARTING LONG RUN PIPELINE"
echo " 1. Base Model Training (Optuna=50)"
echo " 2. Prop Bets Regressor Training"
echo " 3. Value Bets Monte Carlo (Sims=50000)"
echo "==============================================="

# 1. Base Model Training
echo "[1/3] Running training_pipeline.py with 50 Optuna trials..."
python3 training_pipeline.py --optuna-trials 50

# 2. Prop Bets Training
echo "[2/3] Running prop_bets_pipeline.py..."
python3 prop_bets_pipeline.py

# 3. Value Bets
echo "[3/3] Running value_bets.py with 50,000 Monte Carlo Sims..."
python3 value_bets.py --sims 50000

echo "==============================================="
echo " LONG RUN PIPELINE COMPLETE!"
echo "==============================================="
