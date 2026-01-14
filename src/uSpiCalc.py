import os
import math
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
sys.path.append(os.sep.join(os.getcwd().split(os.sep)[:-1]))
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
        return np.float32(better_mean / self.n_scramble), time_death_means

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


