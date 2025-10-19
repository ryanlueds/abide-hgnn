import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

atlas_filename = '/home/ryan/dev/mlg/abide/preprocessing/resources/abide_rois/ho_mask_pad.nii.gz'
roi_coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)
file_path = 'Outputs/cpac/filt_noglobal/rois_ho/NYU_0051076_rois_ho.1D'

node_colors = 'blue'

try:
    time_series = np.loadtxt(file_path)

    if time_series.shape[1] != len(roi_coords):
        raise ValueError(
            f"Data-Atlas mismatch: The data has {time_series.shape[1]} ROIs, "
            f"but the loaded atlas has {len(roi_coords)} ROIs."
        )

    correlation_matrix = np.corrcoef(time_series, rowvar=False)

    view = plotting.plot_connectome(
        adjacency_matrix=correlation_matrix,
        node_coords=roi_coords,
        edge_threshold="95%",
        node_size=15,
        node_color=node_colors,
        edge_kwargs={'linewidth': 0.5, 'alpha': 0.2}
    )

    output_filename = 'fig.png'
    view.savefig(output_filename, dpi=300)

except Exception as e:
    print(f"An error occurred: {e}")
