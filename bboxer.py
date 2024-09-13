import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from .rectangle_fitting2 import LShapeFitting
import matplotlib.axes as mpl_axes
from .utils import *

class Bboxer:
    def __init__(self, z_threshold=2.0):
        self.z_threshold = z_threshold
        # self.lshape_fitting = LShapeFitting()
        self.points_xy = None
        self.points = None
        self.clustered_points = None
        self.clustered_labels = None
        self.rects = None
        
    def initialize(self, points: np.ndarray):
        """
        Attributes:
        - points_xy: A nx3 numpy array of non-ground LiDAR points in the frame
        """
        self.points = points
        self.points_xy = points[:, :2]  # Keep only the x and y coordinates

    def cluster(self, ax: mpl_axes= None, visualize=False):
        """
        Perform DBSCAN clustering on the non-ground points.
        Args:
        - ax(optional): Matplotlib axes for plotting. If not passed, will automatically create one
        - visualize(optional): A boolean flag to visualize the clustered points 
        """
        #xy_points = non_ground[:, :2]  # Assuming non_ground is an Nx3 array
        xy_points = self.points_xy
        # Perform DBSCAN clustering on 2D points
        eps = 0.4
        min_samples = 8
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(xy_points)
        
        self.clustered_points = xy_points
        self.clustered_labels = labels
        if visualize:
            if ax is None:
                clustered_points_viz(xy_points, labels)
            if ax is not None:
                clustered_points_viz(clustered_points=xy_points, clustered_labels=labels, ax=ax)

    def estimate_bboxes(self, ax: mpl_axes.Axes = None, plot=False):
        """
        Esimtate bounding boxes (L-Shape fitting) on the clustered points.
        Args:
        - ax(optional): Matplotlib axes for plotting
        - plot(optional): A boolean flag to plot the L-Shape fitting results
        
        provied ax and plot=True to plot the L-Shape fitting results
        """
        lshape_fitting = LShapeFitting()
        clustered_points = self.clustered_points
        clustered_labels = self.clustered_labels
        unique_labels = np.unique(clustered_labels)

        if plot and (ax is None):
            raise ValueError("Please provide a valid Axes object for plotting.")
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points

            # Extract the points belonging to the current cluster
            mask = clustered_labels == label
            cluster_points_x = clustered_points[mask, 0]
            cluster_points_y = clustered_points[mask, 1]

            # Perform L-Shape fitting on the cluster points
            rects, idsets = lshape_fitting.fitting(cluster_points_x, cluster_points_y)
            self.rects = rects
            # Draw the rectangles on the provided axis
            if plot:
                for rect in rects:
                    rect.plot(ax)  # Pass the Axes object to the plot method
        if plot:        
            # Set aspect ratio and labels if needed
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('L-Shape Fitting on Clusters')
            plot_bev_point_cloud(ax, self.points_xy)
    
    def estimate_bboxes2(self, ax: mpl_axes.Axes = None, z_threshold=2.0, plot=False):
        """
        Estimate bounding (L-shape fitting) on clusters and with filtering based on statistical properties. 
        
        Parameters:
        - ax(optional): Matplotlib axes for plotting
        - z_threshold: Z-score threshold for filtering outliers
        
        Returns:
        - Filtered bounding boxes for the current frame
        """
        lshape_fitting = LShapeFitting()
        points = self.points_xy  
        clustered_points = self.clustered_points
        clustered_labels = self.clustered_labels
        unique_labels = np.unique(clustered_labels)
        all_rects = []

        if plot and (ax is None):
            raise ValueError("Please provide a valid Axes object for plotting.")
        
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

        if plot:
            # Step 4: Plot the filtered bounding boxes
            for rect in filtered_rects:
                rect.plot(ax)  # Draw the filtered bounding boxes on the plot
        self.rects = filtered_rects
        return filtered_rects