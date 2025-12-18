import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from src.utils import *


class NucleatorsProbabilities:
    def __init__(self, cells_XY, cells_death_times, cells_neighbors_list):
        self.XY = cells_XY
        self.TIMES = cells_death_times
        self.neighbors_list = cells_neighbors_list

    def calc_nucleators(self, curr_TOD=0, max_TOD=None):
        def propagated_in_single_time_frame(self_ob: NucleatorsProbabilities, start, end, curr_propagated):
            current_dead_mask = (self_ob.TIMES == start)
            current_alive_mask = (self_ob.TIMES > start)
            next_time_frame_dead_mask = (self_ob.TIMES == end)
            n_cells = len(current_dead_mask)

            nucleators_candidates = np.zeros_like(current_dead_mask)

            # collect all dead cells neighbors
            all_neighbors_of_dead = set()
            for curr_dead_idx, curr_dead_stat in enumerate(current_dead_mask):
                if not curr_dead_stat:
                    continue
                dead_cells_neighbors = self_ob.neighbors_list[curr_dead_idx]
                all_neighbors_of_dead.update(dead_cells_neighbors)

            dead_and_marked = set()
            # dead and in neighbors are propagated
            for nei_of_dead in all_neighbors_of_dead:
                if current_dead_mask[nei_of_dead]:
                    curr_propagated[nei_of_dead] = True
                    dead_and_marked.add(nei_of_dead)
            # all dead cells neighbors that are dead in next time frame are propagated
            for neighbor_of_dead_in_curr_frame in all_neighbors_of_dead:
                is_dead_in_next_frame = next_time_frame_dead_mask[neighbor_of_dead_in_curr_frame]
                if not is_dead_in_next_frame:
                    continue
                curr_propagated[neighbor_of_dead_in_curr_frame] = True
                dead_and_marked.add(neighbor_of_dead_in_curr_frame)

            # go through neighbors of marked as propagated, if neighbor is dead in same frame, it is propagated
            dead_and_marked_cpy = dead_and_marked.copy()
            for marked in dead_and_marked_cpy:
                dead_in_next_frame_neighbors_indexes = self_ob.neighbors_list[marked]
                for nei in dead_in_next_frame_neighbors_indexes:
                    if self_ob.TIMES[nei] == self_ob.TIMES[marked]:
                        curr_propagated[nei] = True
                        dead_and_marked.add(nei)

            # go through dead in next frame, if not marked as propagated, divide into blobs
            blobs = []
            dead_in_next_frame_indexes = np.where(next_time_frame_dead_mask)[0]
            for dead_in_next_idx in dead_in_next_frame_indexes:
                # check if the cell or its neighbors are in one of the blobs
                found_blob = False
                for blob in blobs:
                    dead_in_next_frame_neighbors_indexes = self_ob.neighbors_list[dead_in_next_idx]
                    dead_in_next_frame_dead_neighbors_indexes = set(dead_in_next_frame_neighbors_indexes)
                    dead_in_next_frame_dead_neighbors_indexes = dead_in_next_frame_dead_neighbors_indexes.intersection(set(dead_in_next_frame_indexes))
                    for dead_in_next_frame_neighbor_idx in dead_in_next_frame_dead_neighbors_indexes:
                        if dead_in_next_frame_neighbor_idx in blob:
                            blob.append(dead_in_next_idx)
                            found_blob = True
                # if no blob was found to put the cell in, create a new blob
                if not found_blob:
                    blobs.append([dead_in_next_idx])

            # for each blob, choose a single nucleator and the rest are labeled as propagators.
            for blob in blobs:
                choose_curr_cell = False
                for cell_idx in blob:
                    if not choose_curr_cell:
                        # skip first cell as to be nucleator
                        choose_curr_cell = True
                        continue
                    curr_propagated[cell_idx] = True
            # curr_nucleated_probability = len(blobs) / (len()-len(all_neighbors_of_dead))
            # curr_propagated_probability = len(np.where(np.array(list(curr_propagated.values())))[0]) / len(all_neighbors_of_dead)
            return curr_propagated #, curr_propagated_probability, curr_nucleated_probability

        implicit_temporal_resolution = np.unique(self.TIMES)[1] - np.unique(self.TIMES)[0]
        # curr_TOD = 0
        max_time = self.TIMES.max() - implicit_temporal_resolution if max_TOD is None else max_TOD
        accumulated_all_frames_propagated = {key:False for key in range(len(self.TIMES))}
        accumulated_by_frame_propagated = {}
        accumulated_by_frame_nucleators = {}
        while curr_TOD <= max_time:
            accumulated_all_frames_propagated = propagated_in_single_time_frame(self_ob=self, start=curr_TOD, end=curr_TOD+implicit_temporal_resolution, curr_propagated=accumulated_all_frames_propagated.copy())
            # collect by frame accumulated propagated and nucleators
            curr_accumulated_nucleators_mask, curr_accumulated_propagated_mask = np.array(np.array(list(accumulated_all_frames_propagated.values())) - 1,
                                                                                 dtype=bool), \
                                                                                 np.array(np.array(list(accumulated_all_frames_propagated.values())),
                                                                                 dtype=bool)
            accumulated_by_frame_propagated[curr_TOD] = curr_accumulated_propagated_mask
            accumulated_by_frame_nucleators[curr_TOD] = curr_accumulated_nucleators_mask

            curr_TOD += implicit_temporal_resolution
        # if max_TOD is not None:
        #     times_mask = self.TIMES>curr_TOD+implicit_temporal_resolution
        #     single_frame_prop = np.array(np.array(list(propagated.values()))-1, dtype=bool)
        #     single_frame_prop[times_mask] = False
        #     return single_frame_prop
        nucleators_mask, propagated_mask = np.array(np.array(list(accumulated_all_frames_propagated.values()))-1, dtype=bool),\
                                           np.array(np.array(list(accumulated_all_frames_propagated.values())), dtype=bool)
        return nucleators_mask, propagated_mask, accumulated_by_frame_nucleators, accumulated_by_frame_propagated
