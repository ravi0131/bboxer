import matplotlib.axes as mpl_axes
import numpy as np


def plot_bev_point_cloud(ax: mpl_axes.Axes, clustered_points: np.ndarray):
    # Plot the BEV point cloud
    ax.scatter(clustered_points[:, 0], clustered_points[:, 1], s=1, c='gray', label='Point Cloud')
    ax.legend()
    

import open3d as o3d
import matplotlib.pyplot as plt
def BEV_visualization(points: np.ndarray):
    # Project to XY plane (ignore Z)
    bev_points = points[:, :2]  # Just taking the x and y coordinates

    # Optionally, scale or color points based on their Z values (elevation)
    z_values = points[:, 2]
    colors = plt.cm.viridis((z_values - z_values.min()) / (z_values.max() - z_values.min()))

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(bev_points[:, 0], bev_points[:, 1], c=colors, s=0.1, cmap='viridis')
    plt.colorbar(label='Elevation (Z)')
    plt.axis('equal')
    plt.grid(True)
    plt.title('2D Bird\'s Eye View (BEV) of Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()