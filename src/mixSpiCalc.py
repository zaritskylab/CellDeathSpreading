import os
import math
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from sklearn.preprocessing import LabelEncoder
# sys.path.append("/home/esraan/CellDeathSpreading/src/")
from src.utils import read_experiment_cell_xy_and_death_times
from src.SpiCalc import SpiCalc

class mixSpiCalc(SpiCalc):
    # die_times must be in frames number, dist_threshold is assumed to be in micron
    def __init__(self, XY, die_times, death_mode, temporal_resolution, n_scramble=1000,
                 dist_threshold=100, **kwargs):

        super().__init__( XY, die_times, temporal_resolution, n_scramble,
                dist_threshold, **kwargs)
        le = LabelEncoder()
        self.death_mode = le.fit_transform(death_mode)
        self.create_scramble()
        self.neighbors_difference_death_times = self.get_neighbors_difference_death_times()
        self.original_difference_death_times = self.calc_stat(self.neighbors_difference_death_times[0])
        self.scramble_signficance_95, self.scramble_signficance_98, self.scramble_mean_time_death, self.propagation_index = (0,0,0,-1)
        self.statistic_score, self.all_mean_shuffles_sorted = self.assess_stat()
        self.propagation_index = ((self.scramble_signficance_95 - self.original_difference_death_times) / self.scramble_signficance_95)

    def create_scramble(self):
        self.scrambles = []
        for _ in range(self.n_scramble):
            result = self.die_times.copy()
            for mode in np.unique(self.death_mode):
                mask = self.death_mode == mode
                shuffled_withen = self.die_times[mask].copy()
                np.random.shuffle(shuffled_withen)
                result[mask] = shuffled_withen
            self.scrambles.append(result)

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

