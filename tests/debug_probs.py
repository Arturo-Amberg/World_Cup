import pickle
import pandas as pd
import json

df_ev = pd.read_csv("data/coolbet/model_value_bets.csv")
print(df_ev[df_ev['Match'] == 'Spain vs Cape Verde'])
