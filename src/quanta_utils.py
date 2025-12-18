import os
import sys
import shutil
import math
import warnings
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import *
from enum import Enum
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
sys.path.append("/home/esraan/CellDeathSpreading/src/")
from src.utils import get_experiment_cell_death_times_by_specific_siliding_window,read_experiment_cell_xy_and_death_times

def get_real_distance(cell1_xy, cell2_xy):
    """
    calculte the actual euclidean distance between cells
    Args:
        cell1_xy (tuple): x, y cells location
        cell2_xy (tuple): x, y cells location
    Returns:
        int, float: euclidean distance
    """
    cell1_x, cell1_y = cell1_xy
    cell2_x, cell2_y = cell2_xy
    return ((cell1_x - cell2_x)**2 + (cell1_y - cell2_y)**2)**.5

def get_neighbors(XY, dist_threshold=100, filter_neighbors_by_distance=1, filter_neighbors_by_level=3):
    """get morphological/structural niegbors of each cells based on the parameters given
    Args:
        XY (np.array): np.array of cells location
        dist_threshold (int, optional): in microns. Defaults to 100.
        filter_neighbors_by_distance (int, optional): control filtering by distance (0,1- True, False for filtering by distance or not). Defaults to 1.
        filter_neighbors_by_level (int, optional): control filtering by level, chose niegbors from degree 1, degree 2, degree 3. Defaults to 3.
    Returns:
        list|tuple[list]: according to selected parmaters, list of nieghbors are returned, first degree, second degree, and third degree
    """
    vor = Voronoi(XY)
    neighbors = vor.ridge_points
    neighbors_list = []
    neighbors_list2 = []
    neighbors_list3 = []
    for i in range(len(XY)):
        neighbors_list.append([])
        neighbors_list2.append([])
        neighbors_list3.append([])
    for x in neighbors:
        neighbors_list[x[0]].append(x[1])
        neighbors_list[x[1]].append(x[0])
    for i in range(len(XY)):
        for j in neighbors_list[i]:
            neighbors_list2[i] = list(set(neighbors_list2[i]+neighbors_list[j]))
    for i in range(len(XY)):
        for j in neighbors_list2[i]:
            neighbors_list3[i] = list(set(neighbors_list3[i]+neighbors_list2[j]))
    if filter_neighbors_by_distance==1:
        for i in range(len(XY)):
            neighbors_list[i] = list(filter(lambda x: 0 < get_real_distance(XY[i], XY[x])<dist_threshold, neighbors_list[i]))
            neighbors_list2[i] = list(filter(lambda x: 0 < get_real_distance(XY[i], XY[x])<dist_threshold , neighbors_list2[i]))
            neighbors_list3[i] = list(filter(lambda x: 0 < get_real_distance(XY[i], XY[x])<dist_threshold, neighbors_list3[i]))
    if filter_neighbors_by_level== 1:
        return neighbors_list, [],[]
    elif filter_neighbors_by_level == 2:
        neighbors_list2 = [list(set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list2))]
        return neighbors_list, neighbors_list2, []
    elif filter_neighbors_by_level == 3:
        neighbors_list2 = [list(set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list2))]
        neighbors_list3 = [list(set(neighbors_list3[i]) - set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list3))]
        return neighbors_list, neighbors_list2, neighbors_list3
    return neighbors_list, [list(set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list2))], [list(set(neighbors_list3[i]) - set(neighbors_list2[i]) - set(neighbors_list[i])) for i in range(len(neighbors_list3))]

def get_time_difference(death_times, cell1_idx, cell2_idx):
    """
    calculate the time difference between two cells death times
    Args:
        death_times (np.array): array of cells death times
        cell1_idx (int): index of first cell
        cell2_idx (int): index of second cell
    Returns:
        int, float: time difference in seconds
    """
    return abs(death_times[cell1_idx] - death_times[cell2_idx])

def normalize_death_times(death_times, n_type='median_and_percentile_range'):
    """
    Normalize death times to the range [0, 1].
    Args:
        death_times (np.ndarray): Array of death times.
    n_type (str): Normalization type, can be 'median_and_percentile_range' or 'median_and_iqr'.
    Returns:
        np.ndarray: Normalized death times.
    """        
    death_times_norm = death_times.copy()
    if n_type == 'median_and_percentile_range':
        lower, upper = 5, 95
        median_time = np.median(death_times)
        p_low, p_high = np.percentile(death_times, [lower, upper])
        # death_times = np.clip(death_times, p_low, p_high)
        duration = p_high - p_low
        scale = duration if duration != 0 else 0.000000001  # avoid division by zero
        death_times_norm = (death_times - median_time) / scale
    elif n_type == 'median_and_iqr':
        median_time = np.median(death_times)
        q1, q3 = np.percentile(death_times, [25, 75])
        iqr = q3 - q1
        scale = iqr if iqr != 0 else 0.000000001
        death_times_norm = (death_times - median_time) / scale
    elif n_type == 'onset_and_duration':
        lower, upper = 2.5, 97.5
        p_low, p_high = np.percentile(death_times, [lower, upper])
        death_times = np.clip(death_times, p_low, p_high)
        p_low, p_high = np.percentile(death_times, [0, 100])
        duration = p_high - p_low
        mean_t = np.median(death_times)
        scale = duration if duration != 0 else 0.000000001  # avoid division by zero
        death_times_norm = (death_times- p_low) / scale # (min(death_times) - death_times) / scale
    elif n_type == 'onset_and_iqr':
        lower, upper = 2.5, 97.5
        p_low, p_high = np.percentile(death_times, [lower, upper])
        death_times = np.clip(death_times, p_low, p_high)
        q1, q3 = np.percentile(death_times, [25, 75])
        iqr = q3 - q1
        scale = iqr if iqr != 0 else 0.000000001
        death_times_norm = (death_times - min(death_times)) / scale
    elif n_type == 'z_score':
        lower, upper = 5, 95
        p_low, p_high = np.percentile(death_times, [lower, upper])
        death_times = np.clip(death_times, p_low, p_high)
        mean_t = np.mean(death_times)
        std_t = np.std(death_times)
        scale = std_t if std_t != 0 else 0.000000001
        death_times_norm = (death_times - mean_t) / scale
    elif n_type == 'min_max':
        min_t = min(death_times)
        max_t = max(death_times)
        scale = max_t - min_t if (max_t - min_t) != 0 else 0.000000001
        death_times_norm = (death_times - min_t) / scale
    else:
        raise ValueError("Normalization type must be 'median_and_percentile_range' or 'median_and_iqr'.")
    return death_times_norm

def get_cells_density_corrected(cell_locations, dist_threshold, image_dim):
    """
    Calculate the local density of cells in a given radious.
    Parameters
    ----------
    XY : np.ndarray
        Array of shape (n_cells, 2) containing the x and y coordinates of the cells.
    dist_threshold : Union[int, float]
        Distance threshold for considering cells as neighbors, by default 100
        must be in microns
    image_dim : tuple
        Tuple containing the dimensions of the image (x_min, x_max, y_min, y_max).
        must be in microns
    Returns
    -------
    all_cells_local_density_measurment : list
        List of local density measurements for all cells, given specific radious.   
    """
    try:
        x_min, x_max, y_min, y_max = image_dim
        all_cells_local_density_measurment_normalized = []
        area_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
        for X,Y in cell_locations:
            circle = Point(X, Y).buffer(dist_threshold)
        # Intersect the circle with the area boundary
            effective_area = circle.intersection(area_polygon).area
            all_cells_in_the_radious_for_specific_cell = list(filter(lambda cordination_other: True if get_real_distance((X,Y),cordination_other)<dist_threshold else False,cell_locations))
            count = len(all_cells_in_the_radious_for_specific_cell)
            if effective_area <= 0:
                print("effective_area", effective_area)
            all_cells_local_density_measurment_normalized.append(count/ effective_area if effective_area > 0 else 0)
        return all_cells_local_density_measurment_normalized
    except ZeroDivisionError:
        print("ZeroDivisionError")
        return all_cells_local_density_measurment_normalized