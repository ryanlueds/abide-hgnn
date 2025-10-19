import os
import glob
import yaml
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def create_graph_from_timeseries(ts_file, threshold):
    time_series = np.loadtxt(ts_file)
    # Shape: [num_nodes, num_features] => [num_rois, num_time_steps]
    node_features = torch.tensor(time_series.T, dtype=torch.float)

    correlation_matrix = np.corrcoef(time_series, rowvar=False)
    adj_matrix = correlation_matrix > threshold

    # make sure no funny business
    np.fill_diagonal(adj_matrix, False)

    # adjacency matrix -> edge_index format for PyG
    edge_index = torch.tensor(np.array(np.where(adj_matrix)), dtype=torch.long)
    graph_data = Data(x=node_features, edge_index=edge_index)

    return graph_data

def main():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']
    output_dir = config['output_dir']
    threshold = config['correlation_threshold']

    os.makedirs(output_dir, exist_ok=True)
    
    graph_output_dir = os.path.join(output_dir, 'graphs')
    os.makedirs(graph_output_dir, exist_ok=True)

    ts_files = glob.glob(os.path.join(data_dir, '*.1D'))

    if not ts_files:
        print(f"Warning: No .1D files found in {data_dir}. Confirm steps in README.md were followed haha")
        return

    print(f"Found {len(ts_files)} time series files.")

    for ts_file in tqdm(ts_files, desc="Generating graphs"):
        try:
            graph_data = create_graph_from_timeseries(ts_file, threshold)

            base_name = os.path.basename(ts_file)
            subject_id = base_name.split('_rois')[0]
            output_filename = os.path.join(graph_output_dir, f"{subject_id}_graph.pt")

            torch.save(graph_data, output_filename)

        except Exception as e:
            print(f"Error processing {ts_file}: {e}")

if __name__ == '__main__':
    main()
