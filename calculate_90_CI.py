import pandas as pd
import numpy as np
from scipy import stats




def calculate_90_confidence_interval(extreme_lower, extreme_upper):
    extreme_z_score = stats.norm.ppf(0.999)
    target_z_score = stats.norm.ppf(0.95)
    
    implied_mean = (extreme_lower + extreme_upper) / 2
    implied_std = (extreme_upper - extreme_lower) / (2 * extreme_z_score)
    
    lower_90 = implied_mean - (target_z_score * implied_std)
    upper_90 = implied_mean + (target_z_score * implied_std)

    lower_90 = round(lower_90, 2)
    upper_90 = round(upper_90, 2)
    
    return lower_90, upper_90

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        
        if 'Lower_Bound' not in df.columns or 'Upper_Bound' not in df.columns:
            raise ValueError("CSV must contain 'Lower_Bound' and 'Upper_Bound' columns")
        
        df['90%_CI_Lower'] = 0.0
        df['90%_CI_Upper'] = 0.0
        
        for idx, row in df.iterrows():
            lower_90, upper_90 = calculate_90_confidence_interval(row['Lower_Bound'], row['Upper_Bound'])
            df.at[idx, '90%_CI_Lower'] = lower_90
            df.at[idx, '90%_CI_Upper'] = upper_90
        
        output_file = 'category_counts_with_90_CI.csv'
        df.to_csv(output_file, index=False)
        
        print("Processed", len(df), "categories.")
        print("Generating", output_file)
        
        return df
        
    except Exception as e:
        print("Something went wrong:", e)
        return None

if __name__ == "__main__":
    file_path = "category_counts.csv"
    results = process_csv(file_path)
    
    if results is not None:
        print("Sample results:")
        print(results.head())
        
        means = results[['Lower_Bound', 'Upper_Bound', '90%_CI_Lower', '90%_CI_Upper']].mean()
        print("Summary statistics (means):")
        print(means)