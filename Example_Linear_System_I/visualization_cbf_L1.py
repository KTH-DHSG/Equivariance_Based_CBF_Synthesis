"""

    Implementation of the linear system example from the CBF paper (Example 2 in Section IV-C).

    Script for the visualization of the CBF for the linear system example.

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

###############################################################################################

cbf_file_name = '2025-05-20_14-27-21_linear_system_1.json'
cbf_folder_name = r'Example_Linear_System_I/Data'

color_map = 'viridis'  # Colormap for visualization

# Load the precomputed CBF
cbfModule = CBFmodule()
cbfModule.load(filename=cbf_file_name, folder_name=cbf_folder_name)

X, Y = cbfModule.cbf.domain
cbf_values = cbfModule.cbf.cbf_values

# Normalize function values for colormap
norm = mcolors.Normalize(vmin=-10, vmax=cbf_values.max())
cmap = matplotlib.colormaps[color_map]

# Compute the h-function for each point in the domain (for visualization of constraint)
x_obstacle_grid = np.linspace(cbfModule.domain_lower_bound[0], cbfModule.domain_upper_bound[0], 100)
y_obstacle_grid = np.linspace(cbfModule.domain_lower_bound[1], cbfModule.domain_upper_bound[1], 100)

H_values = np.array([[cbfModule.h([xi, yi]) for xi in x_obstacle_grid] for yi in y_obstacle_grid])

#####################################################################################################
# Visualization

plt.figure(figsize=(4, 3))
# If X and Y are meshgrids and cbf_values is 2D:
pcm = plt.pcolormesh(X, Y, cbf_values, cmap=cmap, norm=norm, shading='auto')
plt.colorbar(pcm, label='CBF value', pad=0.01)

# Add CBF contours
contour_levels = list(range(-12, 11, 4))
contours = plt.contour(X, Y, cbf_values, levels=contour_levels, colors='white', linewidths=1, linestyles='solid')
plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

# Get current axis limits before plotting the line
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Add the line y = -2x in dashed black
x_line = np.linspace(np.min(X), np.max(X), 500)
y_line = -2 * x_line
plt.plot(x_line, y_line, 'k--', label='y = -3x')

# Plot the obstacle
cs = plt.contourf(x_obstacle_grid, y_obstacle_grid, H_values, levels=[-np.inf, 0], hatches = ['//'], colors='none')  # no fill

# plt.contourf(x_obstacle_grid, y_obstacle_grid, H_values, levels=[-np.inf, 0], colors='gray', alpha=0.7)  # plot the obstacle with thicker lines
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