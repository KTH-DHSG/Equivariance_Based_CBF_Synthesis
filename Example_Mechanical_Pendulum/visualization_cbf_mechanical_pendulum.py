"""

    Script for the visualization of the CBF.

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
from scipy.interpolate import griddata

###############################################################################################

cbf_file_name = '2025-06-18_10-21-17_cbf_mechanical_pendulum.json'
cbf_folder_name = r'Example_Mechanical_Pendulum/Data'

color_map = 'viridis'  # Colormap for visualization

# Load the precomputed CBF
cbfModule = CBFmodule()
cbfModule.load(filename=cbf_file_name, folder_name=cbf_folder_name)

X, Y = cbfModule.cbf.domain
cbf_values = cbfModule.cbf.cbf_values

# Normalize function values for colormap
norm = mcolors.Normalize(vmin=np.nanmin(cbf_values), vmax=np.nanmax(cbf_values))
cmap = matplotlib.colormaps[color_map]

# Compute the h-function for each point in the domain (for visualization of constraint)
x_obstacle_grid = np.linspace(cbfModule.domain_lower_bound[0], cbfModule.domain_upper_bound[0], 100)
y_obstacle_grid = np.linspace(cbfModule.domain_lower_bound[1], cbfModule.domain_upper_bound[1], 100)

H_values = np.array([[cbfModule.h([xi, yi]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

#####################################################################################################
# Preprocessing: Replace Nan values by interpolation from non-NaN neighbors

def interpolate_nans(data, method='linear'):
    """
    Interpolate NaN values in a 2D array using neighboring non-NaN values.

    Parameters:
    - data: 2D numpy array with NaNs
    - method: Interpolation method: 'linear', 'nearest', or 'cubic'

    Returns:
    - A new 2D array with NaNs replaced by interpolated values
    """
    data = np.array(data, dtype=np.float64)  # Ensure it's float for NaN support
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    # Mask of valid (non-NaN) values
    mask = ~np.isnan(data)

    # Interpolation
    interpolated = griddata(
        (x[mask], y[mask]),           # Known positions
        data[mask],                   # Known values
        (x[~mask], y[~mask]),         # Positions to interpolate
        method=method
    )

    # Fill in interpolated values
    filled = data.copy()
    filled[~mask] = interpolated

    return filled

cbf_values = interpolate_nans(cbf_values, method='linear')

#####################################################################################################
# Visualization

plt.figure(figsize=(3, 2))
# If X and Y are meshgrids and cbf_values is 2D:
pcm = plt.pcolormesh(X, Y, cbf_values, cmap=cmap, norm=norm, shading='auto')
plt.colorbar(pcm, label='CBF value', pad=0.01)

# Add CBF contours
contour_levels = np.arange(-2, 2.5,0.5)
contours = plt.contour(X, Y, cbf_values, levels=contour_levels, colors='white', linewidths=1, linestyles='solid')
plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

# Get current axis limits before plotting the line
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# draw horizontal and vertical lines through the origin
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

# Plot the obstacle
cs = plt.contourf(x_obstacle_grid, y_obstacle_grid, H_values, levels=[-np.inf, 0], hatches = ['//'], colors='none')  # no fill

plt.contour(x_obstacle_grid, y_obstacle_grid, H_values, levels=[0], colors='k', alpha=1,  linewidths=2)      # plot the boundary of the obstacle with transparency

# Show labels only for every second tick on both axes (e.g., -4, -2, 0, 2, 4)
ax = plt.gca()
xticks = ax.get_xticks()
yticks = ax.get_yticks()

x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
xticks_new = np.arange(np.ceil(x_min/2)*2, np.floor(x_max/2)*2 + 1, 2)
yticks_new = np.arange(np.ceil(y_min/2)*2, np.floor(y_max/2)*2 + 1, 2)

ax.set_xticks(xticks_new)
ax.set_yticks(yticks_new)

# Restore the original axis limits
plt.xlim(xlim)
plt.ylim(ylim)

# Set equal scaling for both axes
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.tight_layout()
plt.show()