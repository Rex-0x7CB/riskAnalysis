import pandas as pd

df = pd.read_csv("category_counts_with_90_CI.csv")
df['average'] = (df['90%_CI_Lower'] + df['90%_CI_Upper']) / 2
if df['Percentage'].dtype == 'object':
    df['Percentage'] = df['Percentage'].str.rstrip('%').astype('float') / 100
df['product'] = df['average'] * df['Percentage']
result = df['product'].sum()

print("Summation of all Losses: ", result) 

    