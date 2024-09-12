import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from rectangle_fitting2 import LShapeFitting
import matplotlib.axes as mpl_axes

def dbscan_cluster_viz_2d(non_ground: np.ndarray):
    # Extract only the x and y coordinates for clustering
    xy_points = non_ground[:, :2]  # Assuming non_ground is an Nx3 array

    # Perform DBSCAN clustering on 2D points
    eps = 0.4
    min_samples = 8
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(xy_points)
    
    return xy_points, labels

def perform_lshape_fitting(ax: mpl_axes.Axes, clustered_points: np.ndarray, clustered_labels: np.ndarray):
    lshape_fitting = LShapeFitting()

    unique_labels = np.unique(clustered_labels)

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points

        # Extract the points belonging to the current cluster
        mask = clustered_labels == label
        cluster_points_x = clustered_points[mask, 0]
        cluster_points_y = clustered_points[mask, 1]

        # Perform L-Shape fitting on the cluster points
        rects, idsets = lshape_fitting.fitting(cluster_points_x, cluster_points_y)

        # Draw the rectangles on the provided axis
        for rect in rects:
            rect.plot(ax)  # Pass the Axes object to the plot method

    # Set aspect ratio and labels if needed
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('L-Shape Fitting on Clusters')

from .utils import calculate_frame_stats, filter_bboxes_by_zscore

def perform_lshape_fitting2(ax: mpl_axes.Axes, clustered_points: np.ndarray, clustered_labels: np.ndarray, points: np.ndarray, z_threshold=2.0):
    """
    Perform L-shape fitting on clusters and filter bounding boxes based on statistical properties.
    
    Parameters:
    - ax: Matplotlib axes for plotting
    - clustered_points: np.ndarray of point coordinates (x, y) in the frame
    - clustered_labels: np.ndarray of cluster labels for the points
    - points: The original LiDAR points (x, y) for point density calculation
    - z_threshold: Z-score threshold for filtering outliers
    
    Returns:
    - Filtered bounding boxes for the current frame
    """
    lshape_fitting = LShapeFitting()
    points = points[:, :2]  # Keep only the x and y coordinates
    unique_labels = np.unique(clustered_labels)
    all_rects = []

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points

        # Extract the points belonging to the current cluster
        mask = clustered_labels == label
        cluster_points_x = clustered_points[mask, 0]
        cluster_points_y = clustered_points[mask, 1]

        # Perform L-Shape fitting on the cluster points
        rects, _ = lshape_fitting.fitting(cluster_points_x, cluster_points_y)
        all_rects.extend(rects)

    for rect in all_rects:
        rect.calc_rect_contour()
        # Step 2: Compute frame statistics (mean and std deviation for area, aspect ratio, and density)
    frame_stats = calculate_frame_stats(all_rects, points)

    # Step 3: Filter the bounding boxes based on Z-scores
    filtered_rects = filter_bboxes_by_zscore(all_rects, points, frame_stats, z_threshold)

    # Step 4: Plot the filtered bounding boxes
    for rect in filtered_rects:
        rect.plot(ax)  # Draw the filtered bounding boxes on the plot

    return filtered_rects