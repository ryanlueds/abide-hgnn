import plotly.graph_objects as go
import os
from typing import List

def save_plot(
    train_metric: List[float],
    test_metric: List[float],
    metric_name: str,
):
    """
    Creates and saves a Plotly chart comparing train and test metrics.
    
    Args:
        epoch_nums (List[int]): List of epoch numbers (x-axis).
        train_metric (List[float]): List of training metric values.
        test_metric (List[float]): List of testing metric values.
        metric_name (str): The name of the metric (e.g., "Loss", "Accuracy").
        file_name (str): The filename to save the plot as (e.g., "loss_plot.png").
        save_dir (str): The directory to save the chart in.
    """
    # Ensure the save directory exists
    folder_name = "mlp"
    save_dir = os.path.join("results", folder_name, 'charts')
    os.makedirs(save_dir, exist_ok=True)
    
    fig = go.Figure()
    epoch_nums = list(range(1, 1 + len(train_metric)))
    
    # Add Train Metric Line
    fig.add_trace(go.Scatter(
        x=epoch_nums, y=train_metric, 
        mode='lines', name=f'Train {metric_name}',
        line=dict(color='blue')
    ))
    
    # Add Test Metric Line
    fig.add_trace(go.Scatter(
        x=epoch_nums, y=test_metric, 
        mode='lines', name=f'Test {metric_name}',
        line=dict(color='red')
    ))
    
    # Update layout with titles
    fig.update_layout(
        title=f'Train vs. Test {metric_name} per Epoch',
        xaxis_title='Epoch',
        yaxis_title=metric_name,
        legend_title='Metric'
    )
    
    # Save the image
    file_name = f"{metric_name}_plot.png"
    save_path = os.path.join(save_dir, file_name)
    try:
        fig.write_image(save_path)
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
        print("Please ensure 'plotly' and 'kaleido' are installed: pip install plotly kaleido")

def save_cv_plot(
    train_stats: dict, # Expected keys: 'median', 'q1', 'q3'
    test_stats: dict,  # Expected keys: 'median', 'q1', 'q3'
    metric_name: str
):
    """
    Creates and saves a Plotly chart for Cross Validation with Median lines and IQR shading.
    """
    # 1. Setup Directory
    folder_name = "mlp"
    save_dir = os.path.join("results", folder_name, 'charts_cv') # New Directory
    os.makedirs(save_dir, exist_ok=True)

    fig = go.Figure()
    
    # We assume lengths are consistent across median/q1/q3
    epoch_nums = list(range(1, 1 + len(train_stats['median'])))
    
    # --- HELPER TO DRAW BAND ---
    def add_shaded_trace(x, y_median, y_q1, y_q3, color_rgb, label_prefix):
        # 1. Upper Bound (Q3) - Invisible line, just sets the ceiling
        fig.add_trace(go.Scatter(
            x=x, y=y_q3,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 2. Lower Bound (Q1) - Fills up to the trace before it (Q3)
        fig.add_trace(go.Scatter(
            x=x, y=y_q1,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba({color_rgb}, 0.2)', # 20% opacity
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 3. Median Line - Solid
        fig.add_trace(go.Scatter(
            x=x, y=y_median,
            mode='lines',
            name=f'{label_prefix} Median',
            line=dict(color=f'rgb({color_rgb})', width=2)
        ))

    # Add Train (Blue: 0, 0, 255)
    add_shaded_trace(
        epoch_nums, 
        train_stats['median'], train_stats['q1'], train_stats['q3'], 
        "0, 0, 255", "Train"
    )

    # Add Test (Red: 255, 0, 0)
    add_shaded_trace(
        epoch_nums, 
        test_stats['median'], test_stats['q1'], test_stats['q3'], 
        "255, 0, 0", "Test"
    )

    # Update layout
    fig.update_layout(
        title=f'CV Aggregated: Train vs. Test {metric_name} (Median ± IQR)',
        xaxis_title='Epoch',
        yaxis_title=metric_name,
        legend_title='Metric',
        template='plotly_white'
    )
    
    # Save
    file_name = f"{metric_name}_cv_plot.png"
    save_path = os.path.join(save_dir, file_name)
    try:
        fig.write_image(save_path)
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")