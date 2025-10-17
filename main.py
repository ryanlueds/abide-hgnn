import numpy as np
from nilearn import datasets, plotting
import matplotlib.pyplot as plt


# TODO: we're using two separate atlases :p fix this
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
atlas_filename = schaefer_atlas.maps
roi_coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)

network_labels = schaefer_atlas.labels[1:]

unique_networks = np.unique(network_labels)
cmap = plt.get_cmap('tab10', len(unique_networks))
network_to_color = {network: cmap(i) for i, network in enumerate(unique_networks)}
node_colors = [network_to_color[label] for label in network_labels]

file_path = 'Outputs/cpac/filt_noglobal/rois_cc200/MaxMun_a_0051607_rois_cc200.1D'

try:
    # time series -> adjacency matrix
    time_series = np.loadtxt(file_path)
    correlation_matrix = np.corrcoef(time_series, rowvar=False)

    view = plotting.plot_connectome(
        adjacency_matrix=correlation_matrix,
        node_coords=roi_coords,
        edge_threshold="90%", 
        node_size=15,
        node_color=node_colors,
        edge_kwargs={'linewidth': 0.5, 'alpha': 0.2} 
    )

    view.savefig('fig.png', dpi=300)
    
    print("saved to fig.png")

except FileNotFoundError:
    print(f"Error: The subject data file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
