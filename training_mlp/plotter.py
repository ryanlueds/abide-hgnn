import plotly.graph_objects as go
import os
from typing import List

def save_plot(
    train_metric: List[float],
    test_metric: List[float],
    metric_name: str,
    save_dir: str = "charts"
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