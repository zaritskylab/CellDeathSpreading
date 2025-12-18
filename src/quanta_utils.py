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
sys.path.append(os.sep.join(os.getcwd().split(os.sep)[:-1]))
from src.utils import get_experiment_cell_death_times_by_specific_siliding_window,read_experiment_cell_xy_and_death_times,get_exp_treatment_type_and_temporal_resolution
from src.uSpiCalc import uSpiCalc

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

def calc_all_experiments_SPI_and_NI_for_landscape(
        exp_name: Union[str, List[str]],
        exps_dir_path: str,
        meta_data_full_file_path: str,
        **kwargs) -> np.array:
    if isinstance(exp_name, list):
        results = {}
        for exp in exp_name:
            res = calc_all_experiments_SPI_and_NI_for_landscape(
                exp_name=exp,
                exps_dir_path=exps_dir_path,
                meta_data_full_file_path=meta_data_full_file_path,
                **kwargs
            )
            results[exp] = res
        return results
    try:
        # print(exp_name)
        exp_full_path = os.path.join(exps_dir_path, exp_name)
        dist_threshold = kwargs.get("dist_threshold", 100)
        dist_in_pixel = kwargs.get("dist_in_pixel", False)
        if dist_in_pixel:
            temp_csv_exract = pd.read_csv(meta_data_full_file_path)
            phys_size_x = temp_csv_exract[temp_csv_exract['File Name'] == exp_name]['PhysicalSizeX'].values[0]
            dist_threshold = dist_threshold * phys_size_x
        exp_treatment, exp_temporal_resolution = get_exp_treatment_type_and_temporal_resolution(exp_file_name=exp_name,
                                                                                                meta_data_file_full_path=meta_data_full_file_path)

        cells_locis, cells_tods = read_experiment_cell_xy_and_death_times(exp_full_path=exp_full_path)

        # norm_spi_values = norm_spi(cells_locis=cells_locis,cells_tods=cells_tods,exp_temporal_resolution=exp_temporal_resolution,exp_treatment=exp_treatment)
        spi, pvalue, dist_avg =\
            calc_experiment_SPI(cells_tods=cells_tods,
                                cells_location=cells_locis,
                                exp_temporal_resolution=exp_temporal_resolution,
                                exp_treatment=exp_treatment,
                                **kwargs,
                                
                                # sliding_time_window_size = sliding_time_window_size,
                                # time_unit=kwargs.get('time_unit', 'minutes'),
                                        # filter_neighbors_by_distance=kwargs.get("filter_neighbors_by_distance", True),
                                        # filter_neighbors_by_level=kwargs.get("filter_neighbors_by_level", 1),
                                        )
        return  spi, pvalue#, dist_avg# ,norm_spi_values[0],
    except FileNotFoundError:
        return (None,None,None)

def calc_experiment_SPI(cells_location: list,
                        cells_tods:list,
                        exp_temporal_resolution:int,
                        exp_treatment,
                        **kwargs) -> tuple :
    cells_tods = get_experiment_cell_death_times_by_specific_siliding_window(cells_times_of_death=cells_tods,sliding_window_size = kwargs.get('sliding_time_window_size',10))
    spi_instance = uSpiCalc(XY=cells_location, die_times=cells_tods, temporal_resolution=exp_temporal_resolution, exp_treatment=exp_treatment, **kwargs)
    return spi_instance.get_uspis(), spi_instance.assess_stat()[0], spi_instance.calc_avg_distance()

def replace_ugly_long_name(name, cell_line = ""):
    lower_case_name = name.lower()
    if "fb" in lower_case_name and "peg" not in lower_case_name:
        if cell_line=="":
            return "FAC&BSO"
        # elif "sgCx43" in cell_line:
        #     return "MCF10A+FB"
        elif "10A" in cell_line:
            return "MCF10A+FB"
        elif "HAP1-920H" in cell_line:
            return "HAP1 920 clone H+FB"
        elif "HAP1 920 clone H" in cell_line:
            return "HAP1 920 clone H+FB"
        elif "HAP1" in cell_line:
            return "HAP1+FB"
        elif "MCF7" in cell_line:
            return "MCF7+FB"
        elif "U937" in cell_line:
            return "U937+FB"
        
        if "dense"  in lower_case_name or "sparse" in lower_case_name:
            print("here")
            return "MCF10A+FAC&BSO **"
        
        return cell_line+"+FB"
    elif "fac" in lower_case_name and "peg" not in lower_case_name:
        if cell_line=="":
            return "FAC&BSO"
        elif "sgCx43" in cell_line:
            return "MCF10A+FB"
        if "dense"  in lower_case_name or "sparse" in lower_case_name:
            print("here")
            return "MCF10A+FAC&BSO **"
        # if "MCF10A sgCx43".lower() in cell_line.lower():
        #     return "MCF10A+FB"
        return cell_line+"+FB"
    elif "peg" in lower_case_name:
        if "peg1450" in lower_case_name:
            peg_type = "PEG1450"
        elif "peg3350" in lower_case_name:
            peg_type = "PEG3350"
        else:
            peg_type = "GCAMP"
        # peg_type = "PEG1450" if "peg1450" or "PEG1450" in lower_case_name else "PEG3350"
        if cell_line=="":
            return "FAC&BSO+"+peg_type
        elif "HAP1-920H" in cell_line:
            return "HAP1+FAC&BSO+"+peg_type
        elif "HAP1" in cell_line:
            return "HAP1+FAC&BSO+"+peg_type
        return cell_line +"FAC&BSO+"+peg_type
    elif "tsz" in lower_case_name:
        if cell_line=="":
            return "TSZ" #"U937+TSZ"
        elif "U937" in cell_line:
            return "U937+TSZ"
        return cell_line + "+TSZ"
    elif "ml162" in lower_case_name:
        if cell_line=="":
            return "ML162"
        elif "HAP1" in cell_line:
            return "HAP1+ML162" 
        elif "MCF10A" in cell_line:
            return "MCF10A+ML162"
        elif "MCF7" in cell_line:
            return "MCF7+ML162"
        return cell_line +"+ML162"
    elif "erastin" in lower_case_name:
        if cell_line=="":
            return "Erastin"
        elif "HAP1" in cell_line:
            return "HAP1+erastin"
        return cell_line +"Erastin"
    
    elif "skt" in lower_case_name:
        if cell_line=="":
            return "TRAIL"
        elif "MCF10A" in cell_line:
            return "MCF10A+TRAIL" #"MCF10A+superkiller TRAIL"
        return cell_line+"TRAIL"
    elif "trail" in lower_case_name:
        if cell_line=="":
            return "TRAIL"
        elif "MCF10A" in cell_line:
            return "MCF10A+TRAIL" #"MCF10A+superkiller TRAIL"
        return cell_line+"TRAIL"
    elif "amsh" in lower_case_name:
        if cell_line=="":
            return "C' dots"
        elif "b16f10" in cell_line:
            return "B16F10+C' dots"
        elif "B16F10" in cell_line:
            return "B16F10+C' dots"
        return cell_line+ "+C' dots"
    elif "h2o2" in lower_case_name:
        if cell_line=="":
            return "H2O2"
        elif "MCF7" in cell_line:
            return "MCF7+H2O2"
        return cell_line + "+H2O2"
    elif "sparse" in lower_case_name or "dense" in lower_case_name:
        return "MCF10A+FAC&BSO **"
    else:
        if cell_line=="":
            return lower_case_name
        return cell_line+"+"+lower_case_name
    
def simple_treatment(name):
    # if "field" in name.lower():
    #     return "FB"
    if "nec" in name.lower():
        return "Necrosis"
    elif "apop" in name.lower():
        return "Apoptosis"
    else:
        if "_A" in name:
            return "MixedSubApop"
        elif "_N" in name:
            return "MixedSubNec"
        return "MixedColony"