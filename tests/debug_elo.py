from src.pipelines.training_pipeline import load_data, load_elos, team_elo
import pandas as pd

df = load_data()
elos = load_elos()
print('Spain ELO:', team_elo('Spain', elos))
print('Cape Verde ELO:', team_elo('Cape Verde', elos))
