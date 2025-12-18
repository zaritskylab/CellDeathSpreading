import sys
import os
import numpy as np
from scipy.spatial import Voronoi
from abc import ABC , abstractmethod
sys.path.append("/home/esraan/CellDeathSpreading/src/")
from src.utils import get_experiment_cell_death_times_by_specific_siliding_window

class SpiCalc(ABC):
    def __init__(self, XY, die_times, temporal_resolution, n_scramble=1000,
                dist_threshold=100, **kwargs):
        """ 
        Median-SPI calculator
        Args:
            XY (np.array): location numpy array of cells
            die_times (np.array): death times of cells
            temporal_resolution (int): temporal resultion if needed to convert times from frames. needs to set time_unit t0 frames.
            n_scramble (int, optional): _description_. Defaults to 1000.
            dist_threshold (int, optional): set distant to the value where neighbors cells are considered for calculation. Defaults to 100 micron. Don't use pixels!
            set filter_neighbors_by_distance, filter_neighbors_by_level, for further control on neighbors thresholding.
        """
        self.XY = XY
        self.temporal_resolution = temporal_resolution
        self.n_scramble = n_scramble
        self.n_instances = len(die_times)
        self.die_times = die_times
        self.dist_threshold = dist_threshold
        self.settle_kwargs_var(kwargs)
        self.neighbors_list, self.neighbors_list2, self.neighbors_list3 = SpiCalc.get_neighbors(self.XY, self.dist_threshold, self.filter_neighbors_by_distance, self.filter_neighbors_by_level)
        # self.create_scramble()
        if self.sliding_time_window_size != self.temporal_resolution:
            self.modify_time_res()
    
    @staticmethod
    def get_real_distance(cell1_xy, cell2_xy):
        """Static method that calculte the actual euclidean distance between cells

        Args:
            cell1_xy (tuple): x, y cells location
            cell2_xy (tuple): x, y cells location
        Returns:
            int, float: euclidean distance
        """
        cell1_x, cell1_y = cell1_xy
        cell2_x, cell2_y = cell2_xy
        return ((cell1_x - cell2_x)**2 + (cell1_y - cell2_y)**2)**.5
    
    @staticmethod
    def get_neighbors(XY, dist_threshold=100, filter_neighbors_by_distance=1, filter_neighbors_by_level=3):
        """get niegbors of each cells based on the parameters given

        Args:
            XY (np.array): np.array of cells location
            dist_threshold (int, optional): in microns. Defaults to 100.
            filter_neighbors_by_distance (int, optional): control filtering by distance (0,1- True, False for filtering by distance or not). Defaults to 1.
            filter_neighbors_by_level (int, optional): control filtering by level, chose niegbors from degree 1, degree 2, degree 3. Defaults to 3.

        Returns:
            _type_: _description_
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
                neighbors_list[i] = list(filter(lambda x: 0<SpiCalc.get_real_distance(XY[i], XY[x])<dist_threshold, neighbors_list[i]))
                neighbors_list2[i] = list(filter(lambda x: 0<SpiCalc.get_real_distance(XY[i], XY[x])<dist_threshold , neighbors_list2[i]))
                neighbors_list3[i] = list(filter(lambda x: 0<SpiCalc.get_real_distance(XY[i], XY[x])<dist_threshold, neighbors_list3[i]))
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

    def modify_time_res(self):
        self.die_times = get_experiment_cell_death_times_by_specific_siliding_window(cells_times_of_death=self.die_times, sliding_window_size = self.sliding_time_window_size)
    
    def settle_kwargs_var(self, kwargs):
        """ Set kwargs acquired parameters as attributes.

        Args:
            kwargs (dict): more parameters to set and control.
        """
        self.sliding_time_window_size = kwargs.get('sliding_time_window_size', self.temporal_resolution)
        self.filter_neighbors_by_distance = kwargs.get('filter_neighbors_by_distance', 1)
        self.filter_neighbors_by_level = kwargs.get('filter_neighbors_by_level', 1)
        self.find_nighbors_each_time = kwargs.get('find_nighbors_each_time', False)
        self.time_unit = kwargs.get('time_unit', 'minutes')
        self.treatment_type = kwargs.get('treatment', '')
        self.statistic_score = 1

    def create_scramble(self):
        """
        create shuffled time of death on the same constant cells location
        """
        self.scrambles = [np.random.permutation(self.die_times) for _ in range(self.n_scramble)]
    
    def get_neighbors_difference_death_times(self):
        """create the full permutation of cell detah time as n_scramble

        Returns:
            list[arrays[int]]: list of as n_scramble expremnt of permuted death time on the population
        """
        TODs_dist = []
        TODs_dist.append(self.get_time_from_neighbors(self.die_times, self.XY))
        for i in range(self.n_scramble):
            TODs_dist.append(self.get_time_from_neighbors(self.scrambles[i], self.XY)) #self.die_times, self.scrambles[i]))
        return TODs_dist

    def get_time_from_neighbors(self, times, XY):
        """get time of death of neighbors

        Args:
            times (np.array): time of death for all cells (could be pernuted, not nessecly observed)
            XY (np.array): cells location, if needed, neigbors can be calculted each time. not optimized

        Returns:
            array: array of TOD's from the chosen niegbors.
        """
        time_diff_from_nighbors_list = []
        if self.filter_neighbors_by_distance==1 or self.filter_neighbors_by_distance==0:
            #TODO: need optimization, in case we are back to shuffiling xy and not times of death - as we calculate the lists each time
            if self.find_nighbors_each_time:
                neighbors_level1, neighbors_level2, neighbors_level3 = SpiCalc.get_neighbors(XY, self.dist_threshold, self.filter_neighbors_by_distance, self.filter_neighbors_by_level)
            else:
                neighbors_level1 = self.neighbors_list
                neighbors_level2 = self.neighbors_list2
                neighbors_level3 = self.neighbors_list3
            for idx in range(self.n_instances):
                time_diff_from_nighbors_list.extend([abs(times[idx] - times[neighbor]) for neighbor in neighbors_level1[idx]])
                if self.filter_neighbors_by_level==1:
                    continue
                time_diff_from_nighbors_list.extend([abs(times[idx] - times[neighbor]) for neighbor in neighbors_level2[idx]])
                if self.filter_neighbors_by_level==2:
                    continue
                time_diff_from_nighbors_list.extend([abs(times[idx] - times[neighbor]) for neighbor in neighbors_level3[idx]])
        else:
            vor = Voronoi(XY)
            neighbors = vor.ridge_points
            for i in range(len(neighbors)):
                time_diff_from_nighbors_list.append(abs(times[neighbors[i][0]] - times[neighbors[i][1]]))
        return np.array(time_diff_from_nighbors_list)

    def find_nucluator(self, level):
        nuc_blobs_identifier = {}
        set_of_all_cells = set()
        for cell_idx, loci in enumerate(self.XY):
            if cell_idx in set_of_all_cells:
                continue
            if level > self.filter_neighbors_by_level:
                raise ValueError("Level don't corspond to the attributet level!")
            elif level == 3:
                level_1, level_2, level_3 = self.neighbors_list[cell_idx], self.neighbors_list2[cell_idx], self.neighbors_list3[cell_idx]
            elif level == 2:
                level_1, level_2, level_3 = self.neighbors_list[cell_idx], self.neighbors_list2[cell_idx],[]
            elif level == 1:
                level_1, level_2, level_3 = self.neighbors_list[cell_idx],[],[]
            all_level_niegbors_in_dist_thr = level_1 + level_2 + level_3
            any_smaller_die_times_neighbors = all([self.die_times[neighbor] >= self.die_times[cell_idx] for neighbor in all_level_niegbors_in_dist_thr])
            if not any_smaller_die_times_neighbors:
                to_remove = [neighbor for neighbor in all_level_niegbors_in_dist_thr if (self.die_times[neighbor] < self.die_times[cell_idx] or neighbor in set_of_all_cells)]
                if len(to_remove)>0:
                    nuc_blobs_identifier[cell_idx]= list(set(all_level_niegbors_in_dist_thr)-set(to_remove))
                    set_of_all_cells.add(cell_idx)
                    set_of_all_cells.update(set(nuc_blobs_identifier.get(cell_idx)))
                else:
                    continue
            else: 
                #the cell is the first to die but might have niegbors that were discovered already
                set_new = set(all_level_niegbors_in_dist_thr)
                set_new = set_new - set_of_all_cells       
                nuc_blobs_identifier[cell_idx] = list(set_new)
                set_of_all_cells.add(cell_idx)
                set_of_all_cells.update(set_new)
        
        self.nuc_leader_and_their_community = nuc_blobs_identifier
        return nuc_blobs_identifier

    def get_stat_score(self):
        return self.statistic_score
    
    def get_death_times(self):
        return self.die_times.copy()
    
    def get_cells_loci(self):
        return self.XY.copy()
    
    def get_tod_dist_from_nucluator(self, level):
        nucleator_blob = self.find_nucluator(level=level)
        list_tod_fron_nucluator = []
        list_distance_from_nucluator = []
        communities = {}
        for key, value in nucleator_blob.items():
            if not communities.get(key):
                communities[key] = {'dist':[], 'tod':[]}
            for cell in value:
                single_cell_tod = self.die_times[cell] - self.die_times[key]
                single_cell_tod = single_cell_tod[0]
                single_cell_dist = SpiCalc.get_real_distance(self.XY[cell], self.XY[key])
                list_distance_from_nucluator.append(single_cell_dist)
                list_tod_fron_nucluator.append(single_cell_tod)
                communities.get(key).get('dist', []).append(single_cell_dist)
                communities.get(key).get('tod', []).append(single_cell_tod)
        return list_tod_fron_nucluator, list_distance_from_nucluator, communities

    @abstractmethod
    def assess_stat(self):
        pass

    @abstractmethod
    def calc_stat(self, dist_for_calc):
        pass