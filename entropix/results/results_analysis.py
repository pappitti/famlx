import os
import json
from typing import List, Dict, Tuple, NamedTuple
import numpy as np
from scipy import stats
from plotly import graph_objects as go
from plotly.subplots import make_subplots


RESULT_DIR = 'entropix/results'

WINDOW_SIZE = 30

METRICS = [
    'attention_entropy',
    'attention_varentropy',
    'logits_entropy',
    'logits_varentropy'
]

### ANALYSIS FUNCTIONS
class WindowStats(NamedTuple):
    """Statistics for a specific window of tokens"""
    start_pos: int
    end_pos: int
    mean: float
    std: float
    q1: float
    median: float
    q3: float
    iqr: float
    thresholds: Dict[str, float]

class WindowedAnalysis(NamedTuple):
    """Analysis of how metrics change across windows"""
    windows: List[WindowStats]
    window_means: List[float]  # trend of means across windows
    window_stds: List[float]   # trend of stds across windows
    stability: float  # how stable are the metrics across windows
    drift_magnitude: float  # how much do metrics drift across windows

def calculate_window_stats(
    values: List[float],
    start_pos: int,
    end_pos: int,
    percentile_ranges: List[Tuple[float, str]] = [
        (5, 'very_low'),
        (25, 'low'),
        (75, 'high'),
        (95, 'very_high')
    ]
) -> WindowStats:
    """Calculate statistics for a window of values"""
    window_values = np.array(values)
    
    # Basic statistics
    q1, median, q3 = np.percentile(window_values, [25, 50, 75])
    iqr = q3 - q1
    
    # Calculate thresholds based on percentiles
    thresholds = {
        name: np.percentile(window_values, pct)
        for pct, name in percentile_ranges
    }
    
    return WindowStats(
        start_pos=start_pos,
        end_pos=end_pos,
        mean=np.mean(window_values),
        std=np.std(window_values),
        q1=q1,
        median=median,
        q3=q3,
        iqr=iqr,
        thresholds=thresholds
    )

def analyze_windows(
    values: List[float],
    window_size: int = WINDOW_SIZE,
    min_window_size: int = 10
) -> WindowedAnalysis:
    """Analyze metric patterns across windows"""
    windows = []
    window_means = []
    window_stds = []
    
    # Calculate stats for each window
    for start in range(0, len(values), window_size):
        end = min(start + window_size, len(values))
        window_values = values[start:end]
        
        if len(window_values) >= min_window_size:
            window_stats = calculate_window_stats(
                window_values,
                start,
                end
            )
            windows.append(window_stats)
            window_means.append(window_stats.mean)
            window_stds.append(window_stats.std)
    
    # Calculate stability (how consistent are the windows)
    stability = 1.0 - stats.variation(window_means) if window_means else 0.0
    
    # Calculate drift (trend in means across windows)
    if len(window_means) > 1:
        positions = np.arange(len(window_means))
        drift_magnitude, _ = np.polyfit(positions, window_means, 1)
    else:
        drift_magnitude = 0.0
    
    return WindowedAnalysis(
        windows=windows,
        window_means=window_means,
        window_stds=window_stds,
        stability=stability,
        drift_magnitude=drift_magnitude
    )

def analyze_model_windows(
    prompt_list: List[Dict[str, List[float]]],
    metric_name: str,
    window_size: int = WINDOW_SIZE
) -> Dict[str, List[float]]:
    """Analyze windowed patterns across all prompts for a model"""
    # Collect all window stats
    all_window_stats = []
    for prompt_data in prompt_list:
        values = prompt_data[metric_name]
        windowed = analyze_windows(values, window_size)
        all_window_stats.append(windowed)
    
    # Calculate aggregate statistics
    aggregate_stats = {
        'mean_stability': np.mean([ws.stability for ws in all_window_stats]),
        'std_stability': np.std([ws.stability for ws in all_window_stats]),
        'mean_drift': np.mean([ws.drift_magnitude for ws in all_window_stats]),
        'std_drift': np.std([ws.drift_magnitude for ws in all_window_stats])
    }
    
    # Calculate global percentile-based thresholds
    all_values = []
    for prompt_data in prompt_list:
        all_values.extend(prompt_data[metric_name])
    
    percentiles = [1, 5, 25, 75, 95, 99]
    threshold_values = np.percentile(all_values, percentiles)
    
    aggregate_stats.update({
        f'p{p}': v for p, v in zip(percentiles, threshold_values)
    })
    
    return aggregate_stats

def analyze_model_all_metrics(
    prompt_list: List[Dict[str, List[float]]],
    window_size: int = WINDOW_SIZE,
    min_window_size: int = 10
) -> Dict[str, Tuple[List[WindowedAnalysis], Dict[str, float]]]:
    """Analyze all metrics for a model's prompts"""
    results = {}
    
    for metric in METRICS:
        # Get windowed analysis for each prompt
        prompt_windows = [
            analyze_windows(
                prompt_data[metric],
                window_size,
                min_window_size
            )
            for prompt_data in prompt_list
        ]
        
        # Get aggregate statistics
        aggregate_stats = analyze_model_windows(
            prompt_list,
            metric,
            window_size
        )
        
        results[metric] = (prompt_windows, aggregate_stats)
    
    return results

### VISUALIZATION 
def create_model_visualization(
    model_name: str,
    metrics_analysis: Dict[str, Tuple[List[WindowedAnalysis], Dict[str, float]]]
) -> go.Figure:
    """Create four-panel visualization for a model"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric.replace('_', ' ').title() for metric in METRICS],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for i, metric in enumerate(METRICS):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        prompt_windows, aggregate_stats = metrics_analysis[metric]
        
        # Plot individual prompt trends
        for windows in prompt_windows:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(windows.window_means))),
                    y=windows.window_means,
                    mode='lines',
                    line=dict(color='rgba(31, 119, 180, 0.1)'),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Calculate and plot average trend
        all_means = [w.window_means for w in prompt_windows]
        all_stds = [w.window_stds for w in prompt_windows]
        
        max_len = max(len(m) for m in all_means)
        padded_means = [m + [np.nan] * (max_len - len(m)) for m in all_means]
        padded_stds = [s + [np.nan] * (max_len - len(s)) for s in all_stds]
        
        mean_trend = np.nanmean(padded_means, axis=0)
        std_trend = np.nanmean(padded_stds, axis=0)
        x_vals = list(range(len(mean_trend)))
        
        # Plot confidence interval
        fig.add_trace(
            go.Scatter(
                x=x_vals + x_vals[::-1],
                y=np.concatenate([
                    mean_trend + std_trend,
                    (mean_trend - std_trend)[::-1]
                ]),
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='±1 std'
            ),
            row=row, col=col
        )
        
        # Plot mean trend
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=mean_trend,
                mode='lines',
                line=dict(color='rgb(31, 119, 180)', width=2),
                name=f'{metric} trend',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add statistics annotation
        stats_text = (
            f"Stability: {aggregate_stats['mean_stability']:.2f} ± {aggregate_stats['std_stability']:.2f}<br>"
            f"Drift: {aggregate_stats['mean_drift']:.2f} ± {aggregate_stats['std_drift']:.2f}<br>"
            f"Thresholds:<br>"
            f"  p5: {aggregate_stats['p5']:.2f}<br>"
            f"  p25: {aggregate_stats['p25']:.2f}<br>"
            f"  p75: {aggregate_stats['p75']:.2f}<br>"
            f"  p95: {aggregate_stats['p95']:.2f}"
        )
        
        fig.add_annotation(
            x=0.95,
            y=0.95,
            text=stats_text,
            showarrow=False,
            xref=f'x{i+1}',
            yref=f'y{i+1}',
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='Window Position', row=row, col=col)
        fig.update_yaxes(title_text='Value', row=row, col=col)
    
    fig.update_layout(
        title_text=f'Metric Analysis for {model_name}',
        height=800,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def analyze_and_visualize_all_models(
    models_data: Dict[str, List[Dict[str, List[float]]]],
    output_dir: str,
    window_size: int = WINDOW_SIZE,
    min_window_size: int = 10
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze all models and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Store aggregate statistics for all models
    all_models_stats = {}
    
    for model_name, prompt_list in models_data.items():
        print(f"Analyzing model: {model_name}")
        
        sanitized_model_name = model_name.replace(' ', '_')
        sanitized_model_name = model_name.replace('/', '_')
        # Analyze all metrics
        metrics_analysis = analyze_model_all_metrics(
            prompt_list,
            window_size,
            min_window_size
        )
        
        # Store aggregate statistics
        all_models_stats[model_name] = {
            metric: analysis[1]  # The aggregate stats
            for metric, analysis in metrics_analysis.items()
        }
        
        # Create and save visualization
        fig = create_model_visualization(model_name, metrics_analysis)
        output_path = os.path.join(output_dir, f"{sanitized_model_name}_analysis.html")
        fig.write_html(output_path)
        print(f"Saved visualization to: {output_path}")
    
    return all_models_stats

def main():
    ### Dictionary to store data organized by model
    data_by_model = {}

    ### Iterate through all files in the results directory
    for filename in os.listdir(RESULT_DIR):
        if filename.startswith('entropy_data') and filename.endswith('.json'):
            file_path = os.path.join(RESULT_DIR, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Extract the required fields
                model = data.get('model')
                tokens = data.get('tokens', [])
                attention_entropy = data.get('attention_entropy',[])
                attention_varentropy = data.get('attention_varentropy', [])
                logits_entropy = data.get('logits_entropy', [])
                logits_varentropy = data.get('logits_varentropy', []) 
                
                if not data_by_model.get(model):
                    # Initialize the model entry in the dictionary
                    data_by_model[model] = []
                
                # Append the extracted data to the corresponding model
                data_by_model[model].append({
                    'tokens': tokens,
                    'attention_entropy': attention_entropy,
                    'attention_varentropy': attention_varentropy,
                    'logits_entropy': logits_entropy,
                    'logits_varentropy': logits_varentropy
                })

    ### Output file to store the organized data
    output_file = 'organized_entropy_data.json'
    output_file_path = os.path.join(RESULT_DIR, output_file)

    ### Write the organized data to the output file
    with open(output_file_path, 'w') as outfile:
        json.dump(data_by_model, outfile, indent=4)

    print(f"Data organized and saved to {output_file}")

    # Run analysis and create visualizations
    stats = analyze_and_visualize_all_models(
        data_by_model,
        output_dir=RESULT_DIR,
        window_size=WINDOW_SIZE,
        min_window_size=10
    )
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    main()