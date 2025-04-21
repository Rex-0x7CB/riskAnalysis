import pandas as pd
import numpy as np
from scipy.stats import uniform

def run_monte_carlo_simulations(csv_file, num_simulations, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['Category', 'Percentage', '90%_CI_Lower', '90%_CI_Upper']
        if not all(col in df.columns for col in required_columns):
            alt_columns = {'Event': 'Category', 
                           'Probability of the event occurring in a year': 'Percentage',
                           'Lower Bound 90 CI': '90%_CI_Lower',
                           'Upper Bound 90 CI': '90%_CI_Upper'}
            df = df.rename(columns={k: v for k, v in alt_columns.items() if k in df.columns})
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    results = df.copy()
    results['Random_Result'] = 0
    
    all_simulation_totals = []
    for sim in range(num_simulations):
        results['Random_Result'] = 0.0
        for idx, row in df.iterrows():
            randomNumber = np.random.random()
            event_occurs = randomNumber < row['Percentage']
            
            if event_occurs:
                loss = uniform.rvs(loc=row['90%_CI_Lower'], scale=row['90%_CI_Upper'] - row['90%_CI_Lower'])
                results.at[idx, 'Random_Result'] = loss
        
        total_loss = results['Random_Result'].sum()
        all_simulation_totals.append(total_loss)
        
        if sim == 0:
            example_results = results.copy()
    
    mean_loss = np.mean(all_simulation_totals)
    median_loss = np.median(all_simulation_totals)
    min_loss = np.min(all_simulation_totals)
    max_loss = np.max(all_simulation_totals)
    std_dev = np.std(all_simulation_totals)
    total_sum = example_results['Random_Result'].sum()

    total_row = pd.DataFrame({
        'Category': ['Total:'],
        'Count' : [None],
        'Percentage': [None],
        '90%_CI_Lower': [None],
        '90%_CI_Upper': [None],
        'Random_Result': [total_sum]
    })
    
    return {
        'example_simulation': example_results,
        'simulation_totals': all_simulation_totals,
        'summary_stats': {
            'mean': mean_loss,
            'median': median_loss,
            'min': min_loss,
            'max': max_loss,
            'std_dev': std_dev
        }
    }

def save_simulation_results(results_dict, output_file='simulation_results.csv', column_rename=True):
    example_results = results_dict['example_simulation']
    formatted_results = example_results.copy()
    
    if column_rename:
        column_mapping = {
            'Category': 'Event',
            'Percentage': 'Probability of the event occurring in a year',
            # 'Lower_Bound': 'Lower Bound of the 90% CI',
            '90%_CI_Lower': 'Lower Bound of the 90% CI',
            # 'Upper_Bound': 'Upper Bound of the 90% CI',
            '90%_CI_Upper': 'Upper Bound of the 90% CI',
            'Random_Result': 'Random Result (zero when the event did not occur)'
        }
        formatted_results = formatted_results.rename(columns=column_mapping)
    
    for col in formatted_results.columns:
        if 'Lower' in col or 'Upper' in col:
            formatted_results[col] = formatted_results[col].apply(
                lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x
            )
        elif 'Random' in col:
            formatted_results[col] = formatted_results[col].apply(
                lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) and x > 0 else "0"
            )
    
    formatted_results.to_csv(output_file, index=False)
    
    return output_file

def main():
    # File path to your CSV
    # csv_file = 'category_counts.csv'
    # csv_file = 'category_counts_with_90_CI.csv'
    csv_file = 'category_counts_with_90_CI_999-extreme.csv'

    numberOfSimulations = 1000
    results = run_monte_carlo_simulations(csv_file, num_simulations=numberOfSimulations)
    
    if results:
        output_file = save_simulation_results(results, 'simulation_results.csv', column_rename=True)
        print(f"Simulation results saved to " + output_file)
        print("Summary Statistics:")
        for stat, value in results['summary_stats'].items():
            print(f"{stat.capitalize()}: ${value:,.2f}")
        
        percentiles = [50, 75, 90, 95, 99]
        print("Risk Exposure at Different Confidence Levels:")
        for p in percentiles:
            value = np.percentile(results['simulation_totals'], p)
            print(f"{p}th Percentile: ${value:,.2f}")
            
        save_all_simulations = True
        
        if save_all_simulations:
            for i in range(numberOfSimulations):
                sim_results = run_monte_carlo_simulations(csv_file, num_simulations=1)
                save_simulation_results(sim_results, f'simulationResults\\simulation_results_{i+1}.csv')
            print("All", numberOfSimulations,"simulation CSV files have been generated.")
    else:
        print("Simulation could not be completed due to data loading issues.")

if __name__ == "__main__":
    main()