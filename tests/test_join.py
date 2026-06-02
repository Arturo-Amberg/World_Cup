import pandas as pd
df1 = pd.read_csv('data/intl_results.csv')
df2 = pd.read_csv('data/statsbomb_match_stats.csv')
df1['date'] = df1['date'].astype(str)
df2['date'] = df2['date'].astype(str)
df1_indexed = df1.set_index(['date', 'home_team'])
df2_indexed = df2.set_index(['date', 'home_team'])
joined = df1_indexed.join(df2_indexed[['home_corners']], how='inner')
print("Join length:", len(joined))
