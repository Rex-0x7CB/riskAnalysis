import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os


from monteCarloSim import run_monte_carlo_simulations, save_simulation_results

def generate_risk_curve(simulation_results, title="Cybersecurity Risk Exposure Curve"):
    sorted_results = sorted(simulation_results)
    total_simulations = len(sorted_results)
    probabilities = [(total_simulations - i) / total_simulations for i in range(total_simulations)]
    percentages = [prob * 100 for prob in probabilities]
    
    min_loss = max(1, min(sorted_results))
    max_loss = max(sorted_results)
    
    log_x = np.logspace(np.log10(min_loss), np.log10(max_loss * 1.1), 1000)
    y_values = [100 * (1 - np.searchsorted(sorted_results, x) / total_simulations) for x in log_x]
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace( go.Scatter(x=log_x, y=y_values, mode='lines', name='Risk Curve', line=dict(color='#33a8c7', width=3), ) )
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="Loss (Dollars)", type="log", tickformat="$,.0f", gridcolor='lightgray', showgrid=True, range=[0, np.log10(max_loss * 1.2)] ),
        yaxis=dict(title="Chance of Loss or Greater", ticksuffix="%", range=[0, 100],  gridcolor='lightgray', showgrid=True),
        height=600,
        width=1000,
        template="plotly_white",
        margin=dict(t=100, b=80, l=80, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_dash_app(csv_file):
    app = dash.Dash(__name__)
    num_simulations = 1000
    results = run_monte_carlo_simulations(csv_file, num_simulations=num_simulations)

    fig = generate_risk_curve(results['simulation_totals'])
    
    percentiles = [50, 75, 90, 95, 99]
    percentile_values = [np.percentile(results['simulation_totals'], p) for p in percentiles]
    
    table_style = {
        'width': '100%', 
        'borderCollapse': 'collapse',
        'textAlign': 'center',
        'margin': '0 auto',
        'fontFamily': 'Arial, sans-serif'
    }
    
    header_style = {
        'backgroundColor': '#f2f2f2',
        'borderBottom': '2px solid black',
        'padding': '10px',
        'fontWeight': 'bold',
        'textAlign': 'center'
    }
    
    cell_style = {
        'padding': '8px',
        'borderBottom': '1px solid #ddd',
        'textAlign': 'center'
    }
    
    percentile_rows = [
        html.Tr([
            html.Th("Confidence Level", style=header_style),
            html.Th("Risk Exposure", style=header_style)
        ])
    ]
    
    for p, val in zip(percentiles, percentile_values):
        percentile_rows.append(html.Tr([
            html.Td(f"{p}th Percentile", style=cell_style),
            html.Td(f"${val:,.2f}", style=cell_style)
        ]))
    
    percentile_table = html.Table(percentile_rows, style=table_style)
    
    stat_rows = [
        html.Tr([
            html.Th("Statistic", style=header_style),
            html.Th("Value", style=header_style)
        ])
    ]
    
    for stat, value in results['summary_stats'].items():
        stat_rows.append(html.Tr([
            html.Td(stat.capitalize(), style=cell_style),
            html.Td(f"${value:,.2f}", style=cell_style)
        ]))
    
    stats_table = html.Table(stat_rows, style=table_style)
    title_style = {
        'textAlign': 'center',
        'fontFamily': 'Arial, sans-serif',
        'marginBottom': '30px',
        'fontWeight': 'bold',
        'color': '#2c3e50'
    }
    
    section_style = {
        'width': '80%',
        'margin': '30px auto',
        'padding': '20px',
        'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
        'backgroundColor': 'white',
        'borderRadius': '5px'
    }
    
    section_title_style = {
        'textAlign': 'center',
        'fontFamily': 'Arial, sans-serif',
        'marginBottom': '20px',
        'fontWeight': 'bold',
        'color': '#2c3e50'
    }
    
    app.layout = html.Div([
        html.H1("Cybersecurity Risk Visualization", style=title_style),
        
        html.Div([
            dcc.Graph(
                id='risk-curve',
                figure=fig,
                style={'height': '600px'}
            )
        ], style=section_style),
        
        html.Div([
            html.H3("Risk Exposure at Different Confidence Levels:", style=section_title_style),
            html.Div(percentile_table, style={'display': 'flex', 'justifyContent': 'center'})
        ], style=section_style),
        
        html.Div([
            html.H3("Summary Statistics:", style=section_title_style),
            html.Div(stats_table, style={'display': 'flex', 'justifyContent': 'center'})
        ], style=section_style)
    ], style={'backgroundColor': '#f5f5f5', 'minHeight': '100vh', 'padding': '20px'})
    
    return app

def main():
    csv_file = 'category_counts_with_90_CI.csv'
    if not os.path.exists(csv_file):
        csv_file = 'category_counts_with_90_CI.csv'
        if not os.path.exists(csv_file):
            print("Warning: Default CSV file", csv_file ,"not found.")
            return
    
    app = create_dash_app(csv_file)
    
    print("Starting Dash server...")
    print("Using file:", csv_file)
    
    app.run(debug=True)

if __name__ == "__main__":
    main()