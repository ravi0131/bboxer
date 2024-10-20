import matplotlib.axes as mpl_axes
import numpy as np
from .rectangle_fitting2 import RectangleData
from typing import List, Dict

def clustered_points_viz(clustered_points: np.ndarray, clustered_labels: np.ndarray, ax: mpl_axes.Axes = None):
    """
    Visualize the clustered points in 2D space.
    Args:
    - clustered_points: A nx2 numpy array of clustered points
    - clustered_labels: A nx1 numpy array of cluster labels for each point
    - ax: Matplotlib Axes object to plot on (optional)
    """
    import matplotlib.pyplot as plt
    # Create a colormap instance
    cmap = plt.get_cmap("tab20")  # Use 'tab20' for more colors
    unique_labels = np.unique(clustered_labels)
    colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]

    # Use provided axes or create new ones
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Visualize the 2D clusters
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points
            color = [0, 0, 0, 1]  # Black color for noise
        mask = (clustered_labels == label)
        ax.scatter(clustered_points[mask, 0], clustered_points[mask, 1], color=color, label=f'Cluster {label}', s=10)

    ax.set_title('2D DBSCAN Clustering')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Adjust legend position and appearance as needed
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    ax.set_aspect('equal')
    # Remove tight_layout and figure adjustments
    # plt.tight_layout()  # Not needed when using ax
    # Do not create new figures or call plt.show() here


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



def filter_bboxes_by_zscore(bboxes: List[RectangleData], points: np.ndarray, frame_stats, z_threshold=2.0) -> List[RectangleData]:
    """
    Filter bounding boxes based on Z-score for area, aspect ratio, and point density.
    
    Parameters:
    - bboxes: List of bounding boxes in the frame
    - points: A nx2 numpy array of LiDAR points in the frame
    - frame_stats: Statistics calculated from the frame (mean, std dev of area, aspect ratio, density)
    - z_threshold: Z-score threshold for filtering (default is 2.0, which keeps boxes within 2 std devs)
    
    Returns:
    - List of filtered bounding boxes
    """
    filtered_bboxes = []
    
    for bbox in bboxes:
        area = bbox.length * bbox.width
        aspect_ratio = max(bbox.length / bbox.width, bbox.width / bbox.length)
        density = calculate_point_density(bbox, points)
        
        # Z-scores for each property
        area_z = (area - frame_stats['area_mean']) / frame_stats['area_std']
        aspect_ratio_z = (aspect_ratio - frame_stats['aspect_ratio_mean']) / frame_stats['aspect_ratio_std']
        density_z = (density - frame_stats['density_mean']) / frame_stats['density_std']
        
        # Check if the Z-scores are within the acceptable range
        if abs(area_z) < z_threshold and abs(aspect_ratio_z) < z_threshold and abs(density_z) < z_threshold:
            filtered_bboxes.append(bbox)
    
    return filtered_bboxes

def calculate_point_density(bbox: List[RectangleData], points: np.ndarray) -> float:
    """
    Calculate the point density inside a bounding box.
    
    Parameters:
    - bbox: The bounding box object (RectangleData)
    - points: A nx2 numpy array of LiDAR points in the bbox
    
    Returns:
    - The density of points (points per square meter) inside the bounding box
    """
    # Filter points that fall inside the bounding box
    inside_points = [p for p in points if bbox.contains(p)]
    
    # Calculate density: number of points per square meter
    area = bbox.length * bbox.width
    density = len(inside_points) / area if area > 0 else 0
    
    return density

def calculate_frame_stats(bboxes: List[RectangleData], points: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics (mean, std) for bounding boxes in a frame.
    
    Parameters:
    - bboxes: List of bounding boxes (RectangleData objects)
    - points: A nx2 numpy array consisting of LiDAR points in the frame
    
    Returns:
    - A dictionary containing statistics for area, aspect ratio, and point density
    """
    areas = np.array([bbox.length * bbox.width for bbox in bboxes])
    aspect_ratios = np.array([max(bbox.length / bbox.width, bbox.width / bbox.length) for bbox in bboxes])
    densities = np.array([calculate_point_density(bbox, points) for bbox in bboxes])
    
    # Calculate mean and standard deviation for each property
    frame_stats = {
        'area_mean': np.mean(areas),
        'area_std': np.std(areas),
        'aspect_ratio_mean': np.mean(aspect_ratios),
        'aspect_ratio_std': np.std(aspect_ratios),
        'density_mean': np.mean(densities),
        'density_std': np.std(densities)
    }
    
    return frame_stats