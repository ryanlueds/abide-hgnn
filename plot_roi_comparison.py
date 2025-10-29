import numpy as np
import matplotlib.pyplot as plt

file_path_noglobal = 'Outputs/cpac/filt_noglobal/rois_ho/NYU_0051076_rois_ho.1D'
file_path_global = 'Outputs/cpac/filt_global/rois_ho/NYU_0051076_rois_ho.1D'

time_series_noglobal = np.loadtxt(file_path_noglobal)
time_series_global = np.loadtxt(file_path_global)

roi_noglobal = time_series_noglobal[:, 0]
roi_global = time_series_global[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(roi_noglobal, label='filt_noglobal')
plt.plot(roi_global, label='filt_global')
plt.title('Comparison of Preprocessing Strategies for a Single ROI')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)

output_filename = 'roi_comparison.png'
plt.savefig(output_filename, dpi=300)

print(f"Plot saved to {output_filename}")
