import pandas as pd

file_path = 'Redball_Quantitative_Analysis_2023_data_Unknown_Removed.csv' 
df = pd.read_csv(file_path)


df['Category'] = df['Category'].str.split('::')
df_exploded = df.explode('Category')

category_counts = df_exploded['Category'].value_counts()
total_counts = category_counts.sum()
print("Total Category Count:", total_counts)

category_counts_df = category_counts.reset_index()
category_counts_df.columns = ['Category', 'Count']
category_counts_df['Percentage'] = ((category_counts_df['Count'] / total_counts)).round(4)
print(category_counts_df)

category_counts_df.to_csv('category_counts.csv', header=True)


