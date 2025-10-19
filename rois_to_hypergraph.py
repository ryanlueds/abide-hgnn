import os
import glob
import yaml
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm

def create_hypergraph_from_timeseries(ts_file, threshold):
    time_series = np.loadtxt(ts_file)
    # Shape: [num_nodes, num_features] => [num_rois, num_time_steps]
    node_features = torch.tensor(time_series.T, dtype=torch.float)

    correlation_matrix = np.corrcoef(time_series, rowvar=False)
    adj_matrix = correlation_matrix > threshold

    # no funny business
    np.fill_diagonal(adj_matrix, False)

    G = nx.from_numpy_array(adj_matrix)

    # find maximal cliques. Note: if a hyperedge E1 is a subset of a hyperedge E2, the hyperedge E1 is ignored
    maximal_cliques = list(nx.find_cliques(G))

    # [node_idx, hyperedge_idx]
    node_indices = []
    hyperedge_indices = []
    for i, clique in enumerate(maximal_cliques):
        node_indices.extend(clique)
        hyperedge_indices.extend([i] * len(clique))
    
    if not node_indices:
        # if no cliques are found, create an empty edge index
        hyperedge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)


    hypergraph_data = Data(x=node_features, edge_index=hyperedge_index)
    hypergraph_data.num_hyperedges = len(maximal_cliques)

    return hypergraph_data

def main():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']
    output_dir = config['output_dir']
    threshold = config['correlation_threshold']

    os.makedirs(output_dir, exist_ok=True)

    hypergraph_output_dir = os.path.join(output_dir, 'hypergraphs')
    os.makedirs(hypergraph_output_dir, exist_ok=True)

    ts_files = glob.glob(os.path.join(data_dir, '*.1D'))

    if not ts_files:
        print(f"Warning: No .1D files found in {data_dir}. Confirm steps in README.md were followed haha")
        return

    print(f"Found {len(ts_files)} time series files.")

    for ts_file in tqdm(ts_files, desc="Generating hypergraphs"):
        try:
            hypergraph_data = create_hypergraph_from_timeseries(ts_file, threshold)

            base_name = os.path.basename(ts_file)
            subject_id = base_name.split('_rois')[0]
            output_filename = os.path.join(hypergraph_output_dir, f"{subject_id}_hypergraph.pt")

            torch.save(hypergraph_data, output_filename)

        except Exception as e:
            print(f"Error processing {ts_file}: {e}")

if __name__ == '__main__':
    main()
