import os
import sys
import shutil
import math
import warnings
from typing import *
from enum import Enum
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import math
import copy
from shapely.geometry import Point, Polygon
import numpy as np
from matplotlib.collections import EllipseCollection,CircleCollection,PatchCollection
import math
import numpy as np

sys.path.append("/home/esraan/CellDeathSpreading/src/")
from src.quanta_utils import get_neighbors,get_real_distance

class SegregationIdx:
    def __init__(self,
                XY: np.ndarray, 
                death_modes: np.ndarray,
                num_permutations : int = 1000, 
                dist_threshold: Union[int, float] = 100,
                filter_neighbors_by_distance: bool = True,
                neighbors_level: int = 1,
                **kwargs
                ): 
        """
        Segregation index calculation for a given XY coordinates and death modes.
        Parameters
        ----------
        XY : np.ndarray
            Array of shape (n_cells, 2) containing the x and y coordinates of the cells.
        die_times : np.ndarray
            Array of shape (n_cells,) containing the time of death for each cell.
        death_modes : np.ndarray
            Array of shape (n_cells,) containing the death mode for each cell.
        n_scramble : int, optional
            Number of scrambles to perform, by default 1000
        dist_threshold : Union[int, float], optional
            Distance threshold for considering cells as neighbors, by default 100
            must be in microns 
        filter_neighbors_by_distance : bool, optional
            Whether to filter neighbors by distance, by default True
        neighbors_level : int, optional
            Level of topological neighbors to consider, by default first degree neighbors
        **kwargs : dict, optional
            Additional keyword arguments. filter_nighbors_by_level is an option, you can use it to add more layers of neighbors
            to the calculation. For example, if you want to add second degree neighbors, you can use filter_neighbors_by_level=2.
        """
        self.XY = XY
        # self.die_times = die_times
        self.death_modes = death_modes
        self.num_permutations = num_permutations
        self.dist_threshold = dist_threshold
        self.stats_to_calculate = kwargs.get('stats_to_calculate', 'mean')
        self.filter_neighbors_by_level = kwargs.get('filter_neighbors_by_level', False)
        self.neighbors_level = neighbors_level
        self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = [], [], []
        if filter_neighbors_by_distance: 
            if self.filter_neighbors_by_level:
                self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.XY, self.dist_threshold, True, neighbors_level)
            else:
                self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.XY, self.dist_threshold, True, 3)
        elif self.filter_neighbors_by_level:
            if not filter_neighbors_by_distance:
                self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.XY, self.dist_threshold, False, neighbors_level)
        else:
            self.neighbors_level_1, self.neighbors_level_2, self.neighbors_level_3 = get_neighbors(self.XY, self.dist_threshold, False, 3)
        #TODO: adjust the nighbors_to_include
        # self.segregation_index = self.calculate_segregation_index(XY= self.XY,
        #                                                     neighbors_to_include=self.neighbors_level_1,
        #                                                     labels=self.death_modes,
        #                                                     dist_threshold = self.dist_threshold,
        #                                                     **kwargs)

    def get_segregation_index(self, **kwargs) -> dict:
        """
        Get the segregation index for the given XY coordinates and death modes.
        Returns
        -------
        dict
            Dictionary containing the segregation index for each cell and the overall segregation index.
        """
        self.segregation_index = self.calculate_segregation_index(XY= self.XY,
                                                            neighbors_to_include=self.neighbors_level_1,
                                                            labels=self.death_modes,
                                                            dist_threshold = self.dist_threshold,
                                                            **kwargs)
        return self.segregation_index
           
    def calculate_segregation_index(self, XY:[list,np.ndarray],
                                    neighbors_to_include: [list,np.ndarray] = None,
                                    labels : Union[list,np.ndarray] = [],
                                    dist_threshold: [int, float] = 100.0,
                                    **kwargs) -> dict:
        """
        Calculate the segregation index for a given XY coordinates and labels.
        Parameters
        ----------
        XY : [list, np.ndarray]
            Array of shape (n_cells, 2) containing the x and y coordinates of the cells.    
        neighbors_to_include : [list, np.ndarray], optional
            List of neighbors to include in the calculation.
            if you provide single level neighbors_to_include, and you intend to perform further use needs_to_filter_by_distance and set it to True. 
            otherwise, no further filtering will be performed.
        labels : Union[list, np.ndarray], optional
            Array of shape (n_cells,) containing the labels for each cell, by default []
        dist_threshold : [int, float], optional
            Distance threshold for considering cells as neighbors, by default 100.0
            must be in microns
        stats_to_calculate : str, optional
            Statistics to calculate, by default 'mean'
        num_permutations : int, optional
            Number of scrambles to perform, by default 1000
        **kwargs : dict, optional
            Additional keyword arguments. needs_to_filter_by_distance is an option to use only if nighbors_to_include is given and distance thresholding is needed.
            
        Returns
        -------
        Get the segregation index for the given XY coordinates and death modes.
        Returns
        -------
        dict
            Dictionary containing the segregation index for each cell and the overall segregation index.
        """
        # Optional: add .lower() for case-insensitivity too
        labels_encoding = {label: idx for idx, label in enumerate(set(labels.flatten().tolist()))}
        reverse_encoded_labels = {idx: label for label, idx in labels_encoding.items()}
        encoded_labels_to_array = np.array([labels_encoding.get(i, -1) for i in labels.flatten().tolist()])
        label_counts = {label: np.sum(encoded_labels_to_array == idx)/len(labels) for label, idx in labels_encoding.items()}
        # np.random.seed(kwargs.get('seed', 2019))
        original_seg_idx = self.calculate_segregation_index_once(XY, neighbors_to_include, encoded_labels_to_array,reverse_encoded_labels, **kwargs)
        permuted_si_list = {key: [] for key in original_seg_idx.keys()}
        for i in range(kwargs.get('num_permutations', 1000)):
            shuffled_labels = np.random.permutation(encoded_labels_to_array) 
            shuffled_seg_idx = self.calculate_segregation_index_once(XY, neighbors_to_include, shuffled_labels,reverse_encoded_labels, **kwargs)
            for key in shuffled_seg_idx.keys():
                permuted_si_list.get(key, []).append(shuffled_seg_idx.get(key))
        observed_seg_idx = original_seg_idx.copy()
        self.permuted_si = permuted_si_list.copy()
        p_value = {key: np.sum(np.array(permuted_si_list.get(key)) > observed_seg_idx.get(key)) for key in permuted_si_list.keys()}
        self.res = {key: (observed_seg_idx.get(key), np.float(p_value.get(key,0.0000)/kwargs.get('num_permutations', 1000)), label_counts.get(key)) for key in observed_seg_idx.keys()}
        # self.res = {key: (observed_seg_idx.get(key), round(p_value.get(key, 0) / kwargs.get('num_permutations', 1000), 3), label_counts.get(key)) for key in observed_seg_idx.keys()}
       
        return self.res

    def calculate_segregation_index_once( self,
                            XY:[list,np.ndarray],
                            neighbors_to_include: [list,np.ndarray] = None,
                            encoded_labels_to_array : Union[list,np.ndarray] = [],
                            reverse_encoded_labels: Union[list,np.ndarray] = [],         
                            **kwargs) -> dict:
        """
        Calculate the segregation index for a given XY coordinates and labels.
        Parameters
        ----------
        XY : [list, np.ndarray]
            Array of shape (n_cells, 2) containing the x and y coordinates of the cells.    
        neighbors_to_include : [list, np.ndarray], optional
            List of neighbors to include in the calculation.
        labels : Union[list, np.ndarray], optional
            Array of shape (n_cells,) containing the labels for each cell, by default []
        **kwargs : dict, optional
            Additional keyword arguments.            
        Returns
        -------
        cells_same_segregation_index : dict
            Dictionary containing the segregation index for each cell and the overall segregation index.
        """
        cells_same_segregation_index = {}
        
        for cell_idx in range(len(XY)):
            cell_neighbors_idxs = neighbors_to_include[cell_idx]
            cell_neighbors_labels = encoded_labels_to_array[cell_neighbors_idxs]
            cell_specific_label = encoded_labels_to_array[cell_idx]
            cells_from_same_label = list(filter(lambda label: label == cell_specific_label, cell_neighbors_labels))
            if not cells_same_segregation_index.get(reverse_encoded_labels.get(cell_specific_label)):
                cells_same_segregation_index[reverse_encoded_labels.get(cell_specific_label)] = []
            cells_same_segregation_index.get(reverse_encoded_labels.get(cell_specific_label)).append(len(cells_from_same_label)/len(cell_neighbors_idxs) if len(cell_neighbors_idxs)!= 0 else 0.000001)

        # seg_idx_dict = {reverse_encoded_labels.get(key): value  for key, value in cells_same_segregation_index.items() if key!= 'all'} 
        # seg_idx_dict['all'] = cells_same_segregation_index['all']

        return {key:np.mean(value) for key, value in cells_same_segregation_index.items()} if kwargs.get('stats_to_calculate', 'mean') == 'mean' else {key: np.median(value) for key, value in cells_same_segregation_index.items()}
  
    # @staticmethod #TODO: fixed the issue with corner and edge cells- tesing is needed
   

if __name__ == "__main__":
    # Example usage
    # OLD DATA
    exps_dir_name = "/sise/assafzar-group/assafzar/Esraa/Others/fully_annotated_data/TimeFrames/"
    meta_data_file_full_path= "/sise/assafzar-group/assafzar/Esraa/Others/ManuallyAnnotatedRoisuu.csv"
    meta_data_extract_exp_names= pd.read_csv(meta_data_file_full_path)
    exp_names = meta_data_extract_exp_names.iloc[:,1]
    print(exp_names[1])
    exp_full_path = os.path.join(exps_dir_name, exp_names[0])
    csv_file = pd.read_csv(exp_full_path)
    dist_threshold = 100
    cells_location = csv_file[["cell_x","cell_y"]].values
    death_modes = csv_file[["Mode"]].values
    # np.random.shuffle(death_modes)
    segregation_index = SegregationIdx(cells_location, death_modes,1000, dist_threshold, filter_neighbors_by_distance=True, neighbors_level=1, stats_to_calculate= "mean")
    print(segregation_index.get_segregation_index())