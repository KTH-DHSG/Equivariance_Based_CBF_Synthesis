"""

       Script for comparing the directly computed CBFs with those CBFs computed based on equivariances.

       Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(str('Predictive_CBF_synthesis_Toolbox'))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from CBF.CBFmodule import CBFmodule
import matplotlib.pyplot as plt

# Parameters
cbf_folder_name = r'Examples_Predicitve_CBF_Toolbox_revisited/B1/Data'
cbf1_direct_file_name = '2025-06-19_00-21-09_b1_1_cbfm_2p8.json'
cbf2_direct_file_name = '2025-06-19_02-52-25_b1_2_cbfm_1p12.json'
cbf1_equiv_file_name = '2025-06-10_16-15-07_b1_1_cbfm_2p8_equi_reduced.json'
cbf2_equiv_file_name = '2025-06-10_16-17-12_b1_2_cbfm_1p12_equi_reduced.json'

orientation_value = 0*np.pi/2  # Orientation angle in radians (first plot)

color_map = 'Greys'  # Colormap for visualization

# Load the precomputed CBF
cbfModule1_direct = CBFmodule()
cbfModule1_direct.load(filename=cbf1_direct_file_name, folder_name=cbf_folder_name)

cbfModule2_direct = CBFmodule()
cbfModule2_direct.load(filename=cbf2_direct_file_name, folder_name=cbf_folder_name)

cbfModule1_equiv = CBFmodule()
cbfModule1_equiv.load(filename=cbf1_equiv_file_name, folder_name=cbf_folder_name)

cbfModule2_equiv = CBFmodule()
cbfModule2_equiv.load(filename=cbf2_equiv_file_name, folder_name=cbf_folder_name)

X1, Y1, PSI1 = cbfModule1_direct.cbf.domain
X2, Y2, PSI2 = cbfModule2_direct.cbf.domain
cbf1_direct_values = cbfModule1_direct.cbf.cbf_values
cbf2_direct_values = cbfModule2_direct.cbf.cbf_values
cbf1_equiv_values = cbfModule1_equiv.cbf.cbf_values
cbf2_equiv_values = cbfModule2_equiv.cbf.cbf_values

# Preprocessing of CBF data for visualization
orientation_index_1 = np.argmin(np.abs(PSI1[0,0,:] - orientation_value))  # Find closest index
orientation_actual_value_1 = PSI1[0,0,orientation_index_1]  # Actual orientation value
orientation_index_2 = np.argmin(np.abs(PSI2[0,0,:] - orientation_value))  # Find closest index
orientation_actual_value_2 = PSI2[0,0,orientation_index_2]
print(f"Plot CBF for the fixed orientation {orientation_actual_value_1} rad")

# Enable interactive mode (only for Python scripts)
plt.ion()

# Extract the 2D slice
cbf1_direct_slice = cbf1_direct_values[:, :, orientation_index_1]
cbf2_direct_slice = cbf2_direct_values[:, :, orientation_index_2]
cbf1_equiv_slice = cbf1_equiv_values[:, :, orientation_index_1]
cbf2_equiv_slice = cbf2_equiv_values[:, :, orientation_index_2]
X1_slice = X1[:, :, orientation_index_1]
Y1_slice = Y1[:, :, orientation_index_1]
X2_slice = X2[:, :, orientation_index_2]
Y2_slice = Y2[:, :, orientation_index_2]

cbf1_diff = cbf1_direct_slice - cbf1_equiv_slice
cbf2_diff = cbf2_direct_slice - cbf2_equiv_slice

# Normalize function values for colormap
norm = mcolors.Normalize(
    vmin=np.nanmin([np.nanmin(cbf1_diff), np.nanmin(cbf2_diff)]),
    vmax=np.nanmax([np.nanmax(cbf1_direct_slice), np.nanmax(cbf2_direct_slice)])
)

cmap = matplotlib.colormaps[color_map]

########################################################################################
# Additional 2D plot for the CBF slice with level curves

fig2, ax2 = plt.subplots(figsize=(8, 6))
# Plot the CBF slice with level curves
contour = ax2.contourf(X1_slice, Y1_slice, cbf1_diff, levels=10, cmap=color_map, norm=norm)

# Add color bar
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label("CBF Value")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"CBF2diff Slice at Orientation = {orientation_actual_value_1} rad (2D View)")
# Display the 2D plot
plt.show()

########################################################################################
# Additional 2D plot for the CBF slice with level curves

fig2, ax2 = plt.subplots(figsize=(8, 6))
# Plot the CBF slice with level curves
contour = ax2.contourf(X2_slice, Y2_slice, cbf2_diff, levels=10, cmap=color_map, norm=norm)

# Add color bar
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label("CBF Value")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"CBF2diff Slice at Orientation = {orientation_actual_value_1} rad (2D View)")
# Display the 2D plot
plt.show()

########################################################################################

# Print max deviation cbf diff 1
print(f"Max deviation CBF1 diff: {np.nanmax(cbf1_diff)}")
print(f"Min deviation CBF1 diff: {np.nanmin(cbf1_diff)}")
# Print max deviation cbf diff 2
print(f"Max deviation CBF2 diff: {np.nanmax(cbf2_diff)}")
print(f"Min deviation CBF2 diff: {np.nanmin(cbf2_diff)}")

print("----------------------------------------------------")

#Elementwise absolute values of the CBF differences
cbf1_diff_abs = np.abs(cbf1_diff)
cbf2_diff_abs = np.abs(cbf2_diff)

# Compute mean and standard deviation of the absolute differences
mean_cbf1_diff_abs = np.nanmean(cbf1_diff_abs)
mean_cbf2_diff_abs = np.nanmean(cbf2_diff_abs)
std_cbf1_diff_abs = np.nanstd(cbf1_diff_abs)
std_cbf2_diff_abs = np.nanstd(cbf2_diff_abs)

print(f"Mean absolute CBF1 diff: {mean_cbf1_diff_abs}")
print(f"Standard deviation absolute CBF1 diff: {std_cbf1_diff_abs}")
print(f"Mean absolute CBF2 diff: {mean_cbf2_diff_abs}")
print(f"Standard deviation absolute CBF2 diff: {std_cbf2_diff_abs}")

input("Press Enter to close plots...")