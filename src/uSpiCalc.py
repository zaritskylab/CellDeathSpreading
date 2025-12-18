import os
import math
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
sys.path.append("/Users/esraan/CodeBase/CellDeathSpreading/")
from utils import read_experiment_cell_xy_and_death_times
from src.SpiCalc import SpiCalc

class uSpiCalc(SpiCalc):
    # die_times must be in frames number, dist_threshold is assumed to be in micron
    def __init__(self, XY, die_times, temporal_resolution, n_scramble=1000,
                 dist_threshold=100, **kwargs):

        super().__init__( XY, die_times, temporal_resolution, n_scramble,
                dist_threshold, **kwargs)
        super().create_scramble()
        self.neighbors_difference_death_times = self.get_neighbors_difference_death_times()
        self.original_difference_death_times = self.calc_stat(self.neighbors_difference_death_times[0])
        self.scramble_signficance_95, self.scramble_signficance_98, self.scramble_mean_time_death, self.propagation_index = (0,0,0,-1)
        self.statistic_score, self.all_mean_shuffles_sorted = self.assess_stat()
        self.propagation_index = ((self.scramble_signficance_95 - self.original_difference_death_times) / self.scramble_signficance_95)

    def get_uspis(self):
        return self.propagation_index

    def assess_stat(self):
        better_mean = 0
        time_death_means = []
        real_mean_time_death = self.calc_stat(self.neighbors_difference_death_times[0])
        if self.time_unit == 'frames':
             self.mean_time_death = real_mean_time_death * self.temporal_resolution
        else:
             self.mean_time_death = real_mean_time_death 
        for i in range(self.n_scramble):
            temp_mean_time_death = self.calc_stat(self.neighbors_difference_death_times[i + 1])
            time_death_means.append(temp_mean_time_death)
            if temp_mean_time_death < real_mean_time_death:
                better_mean += 1
        time_death_means.sort()
        if self.time_unit == 'frames':
            self.scramble_signficance_95 = time_death_means[int(self.n_scramble * 5 / 100)] * self.temporal_resolution
            self.scramble_signficance_98 = time_death_means[int(self.n_scramble * 2 / 100)] * self.temporal_resolution
            self.scramble_mean_time_death = (sum(time_death_means) / len(time_death_means)) * self.temporal_resolution
        else:
            self.scramble_signficance_95 = time_death_means[int(self.n_scramble * 5 / 100)]
            self.scramble_signficance_98 = time_death_means[int(self.n_scramble * 2 / 100)]
            self.scramble_mean_time_death = sum(time_death_means) / len(time_death_means)
        return better_mean / self.n_scramble , time_death_means

    def calc_stat(self, dist_for_calc):
        return np.mean(dist_for_calc)
    
    def calc_avg_distance(self):
        distance_diff_from_nighbors_list = []
        if self.filter_neighbors_by_distance==1 or self.filter_neighbors_by_distance==0:
            neighbors_level1 = self.neighbors_list
            neighbors_level2 = self.neighbors_list2
            neighbors_level3 = self.neighbors_list3
            for idx in range(self.n_instances):
                distance_diff_from_nighbors_list.extend([SpiCalc.get_real_distance(self.XY[idx],self.XY[neighbor]) for neighbor in neighbors_level1[idx]])
                if self.filter_neighbors_by_level==1:
                    continue
                distance_diff_from_nighbors_list.extend([SpiCalc.get_real_distance(self.XY[idx],self.XY[neighbor]) for neighbor in neighbors_level2[idx]])
                if self.filter_neighbors_by_level==2:
                    continue
                distance_diff_from_nighbors_list.extend([SpiCalc.get_real_distance(self.XY[idx],self.XY[neighbor]) for neighbor in neighbors_level3[idx]])
        else:
            vor = Voronoi(self.XY)
            neighbors = vor.ridge_points
            for i in range(len(neighbors)):
                distance_diff_from_nighbors_list.append(0)
        return np.mean(np.array(distance_diff_from_nighbors_list))


if __name__ == '__main__':
    single_exp_full_path = '/sise/assafzar-group/assafzar/Esraa/CellDeathQuantification/Data/Experiments_XYT_CSV/OriginalTimeMinutesData/20160820_10A_FB_xy11.csv'
    cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(exp_full_path=single_exp_full_path)
    ob = uSpiCalc(XY=cells_loci,
                        die_times=cells_times_of_death,
                        treatment='FB',
                        temporal_resolution=30,
                        n_scramble=1000,
                        draw=False,
                        dist_threshold=200, 
                        filter_neighbors_by_distance=True,
                        filter_neighbors_by_level= 3,
                        time_unit='minutes',)
    print(ob.get_uspis())
    print(ob.get_stat_score())
    experiments_dir = '/sise/assafzar-group/assafzar/Esraa/CellDeathQuantification/Data/Experiments_XYT_CSV/OriginalTimeMinutesData/'
    experiment_files = glob.glob(os.path.join(experiments_dir, '*.csv'))
    for single_exp_full_path in experiment_files:
        print(f"Processing file: {single_exp_full_path}")
        cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(exp_full_path=single_exp_full_path)
        ob = uSpiCalc(XY=cells_loci,
                      die_times=cells_times_of_death,
                      treatment='',
                      temporal_resolution=10,
                      n_scramble=1000,
                      draw=False,
                      dist_threshold=200, 
                      filter_neighbors_by_distance=True,
                      filter_neighbors_by_level=3,
                      time_unit='minutes',)
        print(f"Propagation Index (uSpi): {ob.get_uspis()}")
        print(f"Statistic Score: {ob.get_stat_score()}")