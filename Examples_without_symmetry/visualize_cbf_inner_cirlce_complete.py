"""

       Load and visualize a precomputed control barrier function (CBF) for the bicycle model.

       Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from CBF.CBFmodule import CBFmodule
import matplotlib.pyplot as plt

# Parameters
cbf_file_name = '2025-06-17_18-01-41_cbf_inner_circle_complete.json'
cbf_folder_name = r'Examples_without_symmetry/Data'

orientation_value = 0*np.pi  # Orientation angle in radians (first plot)

color_map = 'viridis'  # Colormap for visualization

# Load the precomputed CBF
cbfModule = CBFmodule()
cbfModule.load(filename=cbf_file_name, folder_name=cbf_folder_name)

offset = 4.0

X, Y, PSI = cbfModule.cbf.domain
cbf_values = cbfModule.cbf.cbf_values + offset  # Shifted CBF is still a CBF (see Wiltz et al. 2025)

# Preprocessing of CBF data for visualization
orientation_index = np.argmin(np.abs(PSI[0,0,:] - orientation_value))  # Find closest index
orientation_actual_value = PSI[0,0,orientation_index]  # Actual orientation value
print(f"Plot CBF for the fixed orientation {orientation_actual_value} rad")

# Enable interactive mode (only for Python scripts)
plt.ion()

# Extract the 2D slice
cbf_slice = cbf_values[:, :, orientation_index]
X_slice = X[:, :, orientation_index]
Y_slice = Y[:, :, orientation_index]

# Create mask for where X and Y have entries inside a specified interval
domain_to_plot = [-25, 25, -25, 25] # [x_min, x_max, y_min, y_max]
mask = (X_slice >= domain_to_plot[0]) & (X_slice <= domain_to_plot[1]) & \
       (Y_slice >= domain_to_plot[2]) & (Y_slice <= domain_to_plot[3])

# Apply mask to slice: remove points outside the interval
X_slice = np.ma.masked_array(X_slice, mask=~mask)
Y_slice = np.ma.masked_array(Y_slice, mask=~mask)
cbf_slice = np.ma.masked_array(cbf_slice, mask=~mask)

# Normalize function values for colormap
norm = mcolors.Normalize(vmin=cbf_slice.min(), vmax=cbf_slice.max())
cmap = matplotlib.colormaps[color_map]

# Compute the h-function for each point in the domain (for visualization of constraint)
x_obstacle_grid = np.linspace(cbfModule.domain_lower_bound[0], cbfModule.domain_upper_bound[0], 100)
y_obstacle_grid = np.linspace(cbfModule.domain_lower_bound[1], cbfModule.domain_upper_bound[1], 100)

H_values = np.array([[cbfModule.h([xi, yi]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])
H_values = H_values + offset

# Sort points by depth (from back to front)
X_flat, Y_flat, cbf_flat = X_slice.ravel(), Y_slice.ravel(), cbf_slice.ravel()
depth = X_flat + Y_flat + cbf_flat  # Approximate depth for sorting
sort_idx = np.argsort(depth)  # Sort from farthest to nearest
X_sorted, Y_sorted, cbf_sorted = X_flat[sort_idx], Y_flat[sort_idx], cbf_flat[sort_idx]
colors_sorted = cmap(norm(cbf_sorted))  # Get corresponding colors

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot translucent surface first
ax.plot_surface(X_slice, Y_slice, cbf_slice, cmap=color_map, alpha=1.0, edgecolor='none')

# Plot sorted scatter points second (respects depth occlusion)
ax.scatter(X_sorted, Y_sorted, cbf_sorted, c=cbf_sorted, cmap=color_map, s=30, edgecolor='black', depthshade=True)

# Labels and title
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("CBF Value")
ax.set_xlim(domain_to_plot[0], domain_to_plot[1])
ax.set_ylim(domain_to_plot[2], domain_to_plot[3])
ax.set_title(f"CBF Slice at Orientation = {orientation_actual_value} rad")

# Display the interactive plot
plt.show()

########################################################################################
# Additional 2D plot for the CBF slice with level curves

fig2, ax2 = plt.subplots(figsize=(8, 6))
# Plot the CBF slice with level curves
contour = ax2.contourf(X_slice, Y_slice, cbf_slice, levels=15, cmap=color_map, norm=norm)

# Plot the obstacle
cs = plt.contourf(x_obstacle_grid, y_obstacle_grid, H_values, levels=[-np.inf, 0], hatches = ['//'], colors='none')  # no fill
plt.contour(x_obstacle_grid, y_obstacle_grid, H_values, levels=[0], colors='k', alpha=1,  linewidths=2)      # plot the boundary of the obstacle with transparency

# Add red line along the negative x-axis
x_red = np.linspace(domain_to_plot[0], 0, 500)  # adjust range as needed
y_red = np.zeros_like(x_red)
ax2.plot(x_red, y_red, 'r-', linewidth=2)

# Add color bar
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label("CBF Value")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"CBF Slice at Orientation = {orientation_actual_value} rad (2D View)")
# Display the 2D plot
plt.show()

########################################################################################


input("Press Enter to close plots...")