import json
import os
import shutil
from shutil import copyfile, unpack_archive, make_archive
import random
from typing import *
import numpy as np
import pandas as pd
import sys
from scipy.spatial import Voronoi
from scipy.stats import linregress, zscore, pearsonr
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import silhouette_score as sil_score
from sklearn.metrics import calinski_harabasz_score as cali_h_score
from sklearn.metrics import davies_bouldin_score as davies_b_score
from sklearn.metrics.pairwise import euclidean_distances as euc_dis
from matplotlib.lines import Line2D


def get_custom_legend_artists(labels_to_colors: dict, labels_to_markers: dict):
    """
    creates mpl artists and labels for the legend.
    for each unique marker color combination, a single artists-label pair is created
    :param markers:
    :param colors:
    :param markers_to_labels:
    :return:
    """
    assert len(labels_to_colors) == len(labels_to_markers)

    artists = list()
    labels = list()
    for label in labels_to_colors:
        marker, color = labels_to_markers[label], labels_to_colors[label]
        artists.append(Line2D((0, 0), (0, 1), color=color, marker=marker, linestyle=''))
        labels.append(label)
    return artists, labels


def get_all_possible_mpl_markers():
    """

    :return:
    """
    # todo: add documentation
    possible_markers = Line2D.markers.copy()
    possible_markers.pop('None')
    possible_markers.pop(None)
    possible_markers.pop(' ')
    possible_markers.pop('')
    possible_markers.pop(',')
    return possible_markers


def get_all_unique_treatments(meta_data_file_full_path: str = None):
    """

    :param meta_data_file_full_path:
    :return:
    """
    # todo: add documentation
    if meta_data_file_full_path is None:
        meta_data_file_full_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data',
                                                                                 'Experiments_XYT_CSV',
                                                                                 'Experimentsfig2gh_metadata.csv'])
    meta_data_df = pd.read_csv(meta_data_file_full_path)
    all_treatments = meta_data_df['Treatment'].values
    unique_treatments = np.unique(all_treatments)
    return unique_treatments


def convert_all_digits_in_list_to_ints(list_to_fix: List[str]) -> List:
    """

    :param list_to_fix:
    :return:
    """
    # todo: add documentation
    for element_idx, element in enumerate(list_to_fix):
        if element.isdigit():
            list_to_fix[element_idx] = int(element)
    return list_to_fix


def load_dict_from_json(path: str):
    """

    :param path:
    :return:
    """
    # todo: add documentation
    dict_to_load = None
    try:
        with open(path, 'r') as f:
            dict_to_load = json.load(f)
    except FileNotFoundError as e:
        print(f'problem loading the configuration file at:\n{e}')
    return dict_to_load


def add_values_to_json(path: str, kwargs_to_add: Dict[str, Any]):
    """
    Adds all key and values of kwargs_to_add dictionary to the json file @ path
    :param kwargs_to_add: dictionary
    :param path: str, full path to json file
    :return:
    """
    try:
        current_json_dict = None
        with open(path, 'r') as f:
            current_json_dict = json.load(f)
    except FileNotFoundError as e:
        print(f'problem loading the configuration file at {path}\nEXCEPTION CLAUSE:\n{e}')
    try:
        with open(path, 'w') as f:
            current_json_dict.update(kwargs_to_add)
            json.dump(current_json_dict, f)
    except Exception as e:
        print(f'problem writing the configuration file at {path}\nEXCEPTION CLAUSE:\n{e}')


def write_dict_as_json(path: str, dict_to_write: dict):
    """

    :param path:
    :param dict_to_write:
    :return:
    """
    # todo: add documentation
    try:
        with open(path, 'w') as f:
            json.dump(dict_to_write, f)
    except OSError as e:
        print(f'problem saving the configuration file at:\n{e}')


def get_marker_per_treatment_list(all_treatments: np.array) -> Tuple[List, List, dict, dict]:
    """

    :param all_treatments:
    :return:
    """
    # todo: add documentation
    markers_list_path = os.sep.join([CONFIG_FILES_DIR_PATH, 'markers_list_path.npy'])
    colors_list_path = os.sep.join([CONFIG_FILES_DIR_PATH, 'colors_list_path.npy'])
    treatment_to_marker_dict_path = os.sep.join([CONFIG_FILES_DIR_PATH, 'treatment_to_marker_dict_path.txt'])
    treatment_to_color_dict_path = os.sep.join([CONFIG_FILES_DIR_PATH, 'treatment_to_color_dict_path.txt'])
    # if config files are already loaded, use them for consistent labeling
    if os.path.isfile(markers_list_path):
        markers_list = convert_all_digits_in_list_to_ints(np.load(markers_list_path).tolist())
        colors_list = np.load(colors_list_path).tolist()
        treatment_to_marker_dict = load_dict_from_json(treatment_to_marker_dict_path)
        treatment_to_color_dict = load_dict_from_json(treatment_to_color_dict_path)
        markers_list = [treatment_to_marker_dict[treatment_name] for treatment_name in all_treatments]
        colors_list = [treatment_to_color_dict[treatment_name] for treatment_name in all_treatments]
        return markers_list, colors_list, treatment_to_marker_dict, treatment_to_color_dict

    all_possible_markers = list(get_all_possible_mpl_markers().keys())
    unique_treatments = get_all_unique_treatments()

    treatment_to_marker_dict = {}
    treatment_to_color_dict = {}
    for treatment in unique_treatments:
        rand_marker_idx = np.random.randint(0, len(all_possible_markers))
        # rand_marker_key = list(all_possible_markers.keys())[rand_marker_idx]
        treatment_to_marker_dict[treatment] = all_possible_markers[rand_marker_idx]
        treatment_to_color_dict[treatment] = (random.random(), random.random(), random.random(), 1)
        all_possible_markers.pop(rand_marker_idx)

    markers_list = list()
    colors_list = list()
    # marker_to_treatment_dict = dict()
    # color_to_treatment_dict = dict()
    for treatment in all_treatments:
        marker, color = treatment_to_marker_dict[treatment], treatment_to_color_dict[treatment]
        markers_list.append(marker)
        colors_list.append(color)
        # marker_to_treatment_dict[marker] = treatment
        # color_to_treatment_dict[color] = treatment

    # saving to configuration file
    write_dict_as_json(path=treatment_to_marker_dict_path, dict_to_write=treatment_to_marker_dict)
    write_dict_as_json(path=treatment_to_color_dict_path, dict_to_write=treatment_to_color_dict)
    np.save(markers_list_path, np.array(markers_list))
    np.save(colors_list_path, np.array(colors_list))

    return markers_list, colors_list, treatment_to_marker_dict, treatment_to_color_dict


def get_all_paths_csv_files_in_dir(dir_path: str) -> Tuple[List, List]:
    """
    returns full paths and file names with no file type
    :param dir_path:
    :return:
    """
    if dir_path is None:
        raise ValueError('dir path cant be none')
    full_paths = list(map(lambda x: os.sep.join([dir_path, x]), filter(lambda x: x.endswith('.csv') and
                                                                                 'ds_store' not in x.lower(),
                                                                       os.listdir(dir_path))))
    only_exp_names = list(map(lambda x: x.replace('.csv', ''), filter(lambda x: x.endswith('.csv'),
                                                                      os.listdir(dir_path))))
    return full_paths, only_exp_names


def get_exp_treatment_type_and_temporal_resolution(exp_file_name: str,
                                                   meta_data_file_full_path: str = None,
                                                   compressed_flag: bool = False,
                                                   get_exp_density: bool = False) -> Union[Tuple[str, int],
                                                                                           Tuple[str, int, int]]:
    """
    returns an experiment treatment type and temporal resolution (i.e., interval between frames)
    if get_exp_density is set to True, the function also returns the density of the experiment.
    :param exp_file_name:
    :param meta_data_file_full_path:
    :return:
    """
    if meta_data_file_full_path is None:
        if compressed_flag:
            meta_data_file_full_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data',
                                                                                     'Experiments_XYT_CSV',
                                                                                     'Compressed_Experimentsfig2gh_metadata.csv'])
        else:
            meta_data_file_full_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data',
                                                                                     'Experiments_XYT_CSV',
                                                                                     'Experimentsfig2gh_metadata.csv'])

    meta_data_file = pd.read_csv(meta_data_file_full_path)
    exp_meta_data = meta_data_file[meta_data_file['File Name'] == exp_file_name]
    exp_treatment, exp_time_res = exp_meta_data['Treatment'].values[0], \
                                  int(exp_meta_data['Time Interval (min)'].values[0])
    if get_exp_density:
        exp_density = exp_meta_data['Density(#Cells)'].values[0]
        return exp_treatment, exp_time_res, exp_density

    return exp_treatment, exp_time_res


def read_experiment_cell_xy_and_death_times(exp_full_path: str, need_sorting: bool = False) -> Tuple[np.array, np.array]:
    """
    reads an experiment's csv file, returns the cell loci and times of deaths
    :param exp_full_path:
    :return: Tuple[np.array, np.array] - cells_loci, cells_times_of_death
    """
    full_df = pd.read_csv(exp_full_path)
    if need_sorting:
        full_df.sort_values(by='death_time', inplace=True)
        full_df.reset_index(drop=True, inplace=True)
    cells_loci = full_df.loc[:, ['cell_x', 'cell_y']].values
    cells_times_of_death = full_df.loc[:, ['death_time']].values
    return cells_loci, cells_times_of_death

def round_up_to_multiple(arr: np.ndarray, multiple: int) -> np.ndarray:
    """
    Rounds up the elements of a 1D numpy array to the nearest multiple of a specified number.

    Parameters:
        arr (np.ndarray): Input 1D numpy array.
        multiple (int): The number to which elements should be rounded up.

    Returns:
        np.ndarray: A new array with elements rounded up to the nearest multiple of the specified number.
    """
    if multiple <= 0:
        raise ValueError("The multiple must be a positive integer.")
    return np.ceil(arr / multiple) * multiple

def get_experiment_cell_death_times_by_specific_siliding_window(cells_times_of_death:np.ndarray,  sliding_window_size: int, sliding_window_in_minute: bool = True, **kwargs) ->  Tuple[np.array, np.array]:
    """
    reads an experiment's csv file, returns the cell loci and times of deaths
    :param exp_full_path:
    :return: Tuple[np.array, np.array] - cells_loci, cells_times_of_death manipulated by the sliding window - cells that die in 5 minutes are considered dead in the first 10 minutes
    """
    if not sliding_window_in_minute:
        raise ValueError('sliding_window_in_minute must be True')
    else:
        max_time = cells_times_of_death.max()
        cells_times_of_death = round_up_to_multiple(cells_times_of_death, sliding_window_size)

    return cells_times_of_death

def kl_divergence(p: np.array, q: np.asarray) -> np.ndarray:
    """
    returns the kl divergence score for p and q
    :param p: np array - signal from distribution a
    :param q: np array - signal from distribution b
    :return: float
    """
    if type(p) is not type(np.array) or type(q) is not type(np.array):
        p = np.array(p)
        q = np.array(q)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_euclidean_distance_between_cells_in_pixels(cell1_xy: Union[Tuple, np.array], cell2_xy: Union[Tuple, np.array]) \
        -> Union[float, np.array]:
    """
    returns the real distance
    :param cell1_xy:
    :param cell2_xy:
    :return:
    """
    if isinstance(cell1_xy, np.ndarray):
        if len(cell1_xy.shape) > 1:
            seeder = cell2_xy
            neighbors = cell1_xy
        elif len(cell2_xy.shape) > 1:
            seeder = cell1_xy
            neighbors = cell2_xy
        else:
            return get_euclidean_distance_between_cells_in_pixels(tuple(cell1_xy), tuple(cell2_xy))

        seeder_x, seeder_y = seeder[0], seeder[1]
        neighbors_x, neighbors_y = neighbors[:, 0], neighbors[:, 1]

        return ((seeder_x - neighbors_x) ** 2 + (seeder_y - neighbors_y) ** 2) ** .5

    else:
        cell1_x, cell1_y = cell1_xy
        cell2_x, cell2_y = cell2_xy
        return ((cell1_x - cell2_x) ** 2 + (cell1_y - cell2_y) ** 2) ** .5


def get_linear_regression_line_between_two_signals(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    """
    calculates the linear regression line between two signals and returns the x and y of the new regression line
    :param x:
    :param y:
    :return:
    """
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    x_new = np.linspace(0, x.max(), len(y))
    y_new = model.predict(x_new[:, np.newaxis])
    return x_new, y_new


def get_cells_neighbors(XY, threshold_dist: Union[int, float] = None) -> Tuple[List[int], List[int], List[int]]:
    """
    returns 3 levels of topological neighbors for each cell.
    if threshold_dist is not None, a distance constraint is employed on the neighbors to prevent neighbors that are
    very far away.
    :param XY:
    :param threshold_dist:
    :return:
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
        if threshold_dist is None:
            neighbors_list[x[0]].append(x[1])
            neighbors_list[x[1]].append(x[0])
        else:
            if get_euclidean_distance_between_cells_in_pixels(XY[x[0]], XY[x[1]]) <= threshold_dist:
                neighbors_list[x[0]].append(x[1])
                neighbors_list[x[1]].append(x[0])

    for i in range(len(XY)):
        for j in neighbors_list[i]:
            # add the neighbors of a neighbor (j), exclude yourself (i) and
            # remove duplicates and cells that are in 1st neighbor level lists to i
            combined_neighbors = set(neighbors_list2[i] + neighbors_list[j])
            combined_neighbors.remove(i)
            clean_neighbors = combined_neighbors.copy()
            for neighbor_idx in combined_neighbors: clean_neighbors.remove(neighbor_idx) if neighbor_idx in \
                                                                                            neighbors_list[
                                                                                                i] else None
            neighbors_list2[i] = list(clean_neighbors)
    for i in range(len(XY)):
        for j in neighbors_list2[i]:
            # add the 2nd level neighbors of a neighbor (j), exclude yourself (i) and
            # remove duplicates and cells that are in 2nd and 1st neighbor level lists to i
            combined_neighbors = set(neighbors_list3[i] + neighbors_list2[j])
            combined_neighbors.remove(i)
            clean_neighbors = combined_neighbors.copy()
            for neighbor_idx in combined_neighbors: clean_neighbors.remove(neighbor_idx) if neighbor_idx in \
                                                                                            neighbors_list2[
                                                                                                i] else None
            for neighbor_idx in combined_neighbors: clean_neighbors.remove(neighbor_idx) if neighbor_idx in \
                                                                                            neighbors_list[
                                                                                                i] else None
            neighbors_list3[i] = list(clean_neighbors)

    return neighbors_list, neighbors_list2, neighbors_list3


def calc_fraction_from_candidates(dead_cells_at_time_indices: np.array, candidates_indices: np.array) -> float:
    """
    calculates the fraction of cells that died out of the candidates of the specific type of death.
    if there are no candidates, verifies that there are no dead cells and returns 0.
    :param dead_cells_at_time_indices:
    :param candidates_indices:
    :return:
    """
    if len(dead_cells_at_time_indices) > len(candidates_indices):
        raise ValueError('candidates number cant be less than dead cells')
    if len(candidates_indices) == 0:
        return 0
    return len(dead_cells_at_time_indices) / len(candidates_indices)


def calc_mask_from_indices(empty_mask: np.array, indices: Union[np.array, List], val_to_mask: bool = True) -> np.array:
    """
    for each idx in indices list, change the value @idx in the empty mask to val_to_mask
    :param empty_mask:
    :param indices:
    :param val_to_mask:
    :return:
    """
    for idx in indices:
        empty_mask[idx] = val_to_mask
    return empty_mask


def get_cells_not_neighboring_dead_cells(dead_cells_mask, neighbors, neighbors_list2, neighbors_list3, xy=None,
                                         threshold=200):
    """
    returns two groups of cells. 1st is all alive cells that are neighbors of dead cells' neighbors.
    2nd is the rest of the alive cells which are not direct topological neighbors of any dead cells.
    :param dead_cells_mask:
    :param neighbors:
    :param neighbors_list2:
    :param xy:
    :param threshold:
    :return:
    """
    all_alive_cells = np.array(dead_cells_mask - 1, dtype=bool)
    # get all cells neighboring dead cells (propagation candidates)
    around_dead_cells = np.zeros(dead_cells_mask.shape, dtype=bool)
    for cell_idx, is_dead in enumerate(dead_cells_mask):
        if is_dead:
            curr_neighbors = neighbors[cell_idx]
            for neighbor_idx in curr_neighbors:
                if xy is not None:
                    dist = get_euclidean_distance_between_cells_in_pixels(cell1_xy=xy[cell_idx],
                                                                          cell2_xy=xy[neighbor_idx])
                    around_dead_cells[neighbor_idx] = (True) * (dist < threshold)

    # get complementary & alive cells that are not near dead cells
    all_not_around_dead_cells_and_alive = np.array(around_dead_cells - 1, dtype=bool) * all_alive_cells
    # divide to two groups at different "neighboring" distances
    not_around_dead_cells_1 = np.zeros(dead_cells_mask.shape, dtype=bool)
    not_around_dead_cells_2 = np.zeros(dead_cells_mask.shape, dtype=bool)
    for cell_idx, is_cell_not_adjacent_to_death in enumerate(all_not_around_dead_cells_and_alive):
        if is_cell_not_adjacent_to_death:
            alive_cell_2nd_lvl_neighbors = neighbors_list2[cell_idx]
            for adjacent_neighbor_idx in alive_cell_2nd_lvl_neighbors:
                # if the cell(cell_idx) is a 2nd lvl neighbor to a dead cell
                if dead_cells_mask[adjacent_neighbor_idx]:
                    not_around_dead_cells_1[cell_idx] = True
                    break

            alive_cell_3rd_lvl_neighbors = neighbors_list3[cell_idx]
            for adjacent_neighbor_idx in alive_cell_3rd_lvl_neighbors:
                # if the cell(cell_idx) is a 3rd lvl neighbor to a dead cell
                # and not a 2nd lvl neighbor to a dead cell
                if dead_cells_mask[adjacent_neighbor_idx]:
                    not_around_dead_cells_2[cell_idx] = True * (not not_around_dead_cells_1[cell_idx])
                    break

    # not_around_dead_cells_2 = np.array(not_around_dead_cells_1-1, dtype=bool) * all_alive_cells
    return not_around_dead_cells_1, not_around_dead_cells_2


def calc_distance_metric_between_signals(y_true: np.array, y_pred: np.array, metric: str = 'rmse'):
    """
    calculates a distance metric between two np.array values, enforces equal lengths of arrays.
    supports lists, tuples and any iterables as well.
    9/08/2021 - supports the following metrics: rmse, mse, kl-divergence, euclidean distance (returns distances mean).
    :param y_true: np.array
    :param y_pred: np.array
    :param metric: str, metric to calculate
    :return: float, the metric calculation result.
    """
    assert len(y_true) == len(y_pred), f'y_true and y_pred must have equal lengths but y_true length = {len(y_true)}' \
                                       f' and y_pred length = {len(y_pred)}'
    if metric == 'rmse':
        return mse(y_true=y_true, y_pred=y_pred, squared=False)
    if metric == 'mse':
        return mse(y_true=y_true, y_pred=y_pred, squared=True)
    if metric == 'kl_divergence':
        return kl_divergence(y_true, y_pred)
    if metric == 'euclidean':
        return euc_dis(y_true, y_pred).mean()


def calc_signal_slope_and_intercept(x: np.array = None, y: np.array = None) -> Tuple[float, float]:
    """
    calculates a signal slope and intercept using scipy linegress model.
    if x is not given, this function generates an array of consequential indices with an interval of 1 and
    uses it as the signal x-axis.
    the function returns the slope and intercept attributes of the calculated linegress object.
    :param x: np.array - the signal x-axis
    :param y: np.array - the signal values (y-axis)
    :return: Tuple[float,float], slope and intercept accordingly
    """
    assert y is not None, 'Y cant be None!'
    if x is None:
        x = np.arange(0, len(y), 1)

    lr_object = linregress(x, y)
    return lr_object.slope, lr_object.intercept


def clean_string_from_bad_chars(treatment_name: str, replacement='_') -> str:
    return treatment_name.replace('\\', replacement).replace('/', replacement)


def normalize(values: np.array, normalization_method: str = 'z_score', axis: int = 0):
    values = np.array(values)
    if normalization_method == 'z_score':
        return zscore(values, axis=axis)
    if normalization_method == 'min_max':
        max_val = values.max()
        min_val = values.min()
        return (max_val - values) / (max_val - min_val)

    Warning('the normalization method is unknown!')
    return values


def get_dead_cells_mask_in_window(window_start_time_min: int,
                                  window_end_time_min: int,
                                  cells_times_of_death: np.array,
                                  consider_death_in_window_only) -> np.array:
    """
    returns a boolean np.array the shape of cells_times_of_death where each value is True if the cell is dead
    and False otherwise.
    :param window_start_time_min: int, the starting time of the window in minutes
    :param window_end_time_min: int, the end time of the window in minutes
    :param cells_times_of_death: np.array, cells' times of death (in minutes) indexed according to cell indices.
    :param consider_death_in_window_only: bool, whether to consider death which occurred prior to window_start_time_min.
    :return:
    """
    if consider_death_in_window_only:
        return (cells_times_of_death >= window_start_time_min) * (cells_times_of_death < window_end_time_min)
    else:
        return cells_times_of_death < window_end_time_min


def verify_any_str_from_lst_in_specific_str(str_to_verify: str, lst_of_strings: List[str]) -> bool:
    """
    checks if any string from a list of strings appears in another string.
    Used to check whether any of a list of shortened/lazy versions of treatments' names appear in a treatment full name.
    The purpose of this function is to skip un-wanted treatments.
    :param str_to_verify: str, the full string we we like to find instances of at least one of the strings in lst_of_strings
    :param lst_of_strings: List[str], a list of shortened versions of strings.
    :return: boolean
    """
    return sum(
        [treatment_shortname.lower() in str_to_verify.lower() for treatment_shortname in lst_of_strings]) == 0


def calc_correlation(x: np.array, y: np.array, type_of_correlation: str = None,
                     print_p_val: bool = True, return_p_val: bool = False, **kwargs) -> Union[float, np.array]:
    """
    default is pearson correlation (if type_of_correlation argument is None)
    :param x: np.array
    :param y: np.array
    :param type_of_correlation: str
    :param print_p_val: bool
    :param kwargs: any kwargs for correlation functions of scipy
    :return:
    """
    p_val = None

    if type_of_correlation == 'discrete_correlation':
        correlation = correlate(np.array(x).flatten(), np.array(y).flatten(),
                                mode=kwargs.get('correlate_mode', 'full'),
                                method=kwargs.get('correlate_method', 'auto'))
    else:
        Warning('Pearson correlation assumes normal distribution of x and y')
        correlation, p_val = pearsonr(np.array(x).flatten(), np.array(y).flatten())

    if print_p_val:
        print(f'p value of {type_of_correlation}={p_val}')

    if return_p_val:
        return {'correlation': correlation, 'p_val': p_val}
    return correlation


def clusters_evaluation(x_values: np.array,
                        y_values: np.array,
                        clustering_labels: np.array,
                        cluster_evaluation_method: str = 'silhouette_coefficient',
                        **kwargs) -> Union[Tuple[Dict[str, float], float], float]:
    """
    this function calculates the clustering performance of some clustering algorithm,
        ALL scoring measures are intrinsic - do not require "ground-truth" labels, assuming the clustering was correct.
    for each x,y coordinate of a specific datapoint (zipped by order), the cluster label is the corresponding cluster
        label in 'clustering_labels' parameter.
        e.g. : given the values: x_values = [0, 1, 2] | y_values = [3, 4, 5] | clustering_labels = ['0', '1', '1']
            the cluster '1' contains the points (1, 4) and (2, 5) and the cluster '0' contains the point (0, 3).

    possible methods to evaluate clustering performance:
        All clusters evaluations scores combined (returning a single float value):
             silhouette_coefficient -  (b - a)/max(a, b)
                    The Silhouette Coefficient is defined for each sample and is composed of two scores:
                    a: The mean distance between a sample and all other points in the same cluster.
                    b: The mean distance between a sample and all other points in the next nearest cluster.
                    Finally, the mean coefficient is calculated and returned
             calinski_harabasz_index - The index is the ratio of the sum of between-clusters dispersion and of
                    within-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared)

             davies_bouldin_index - This index signifies the average ‘similarity’ between clusters, where the similarity
                    is a measure that compares the distance between clusters with the size of the clusters themselves.
                    Zero is the lowest possible score. Values closer to zero indicate a better partition.

             for all methods pros and cons,
                go to sklearn.metrics documentation: https://scikit-learn.org/stable/modules/clustering.html

        By cluster evaluation and scores (returning by cluster scores dictionary and a mean score for all clusters):
            NOT IMPLEMENTED YET - raises NotImplemented exception.

    :param x_values:
    :param y_values:
    :param clustering_labels:
    :param cluster_evaluation_method:
    :return:
    """
    if cluster_evaluation_method == 'silhouette_coefficient':
        metric = kwargs.get('silhouette_coefficient_metric', 'euclidean')
        cluster_eval_score = sil_score(np.hstack((x_values.reshape(-1, 1), np.array(y_values).reshape(-1, 1))),
                                       clustering_labels, metric=metric)
        return cluster_eval_score
    elif cluster_evaluation_method == 'calinski_harabasz_index':
        cluster_eval_score = cali_h_score(np.hstack((x_values.reshape(-1, 1), np.array(y_values).reshape(-1, 1))),
                                          clustering_labels)
        return cluster_eval_score
    elif cluster_evaluation_method == 'davies_bouldin_index':
        cluster_eval_score = davies_b_score(np.hstack((x_values.reshape(-1, 1), np.array(y_values).reshape(-1, 1))),
                                            clustering_labels)
        return cluster_eval_score

    # by cluster evaluation:
    else:
        # sort x and y values by clusters:
        clusters = {}
        for x, y, label in zip(x_values, y_values, clustering_labels):
            clusters[label] = clusters.get(label, []) + [x, y]
        # turn list of x y coordinates into np.arrays
        for key, val in clusters.items():
            clusters[key] = np.array(val)

        clusters_metric_scores = {}
        for label, cluster in clusters.items():
            cluster_eval_score = 0
            if cluster_evaluation_method == 'silhouette_coefficient':
                metric = kwargs.get('silhouette_coefficient_metric', 'euclidean')
                # cluster_eval_score =
                raise NotImplemented(f'The {cluster_evaluation_method} clustering evaluation method is not '
                                     f'implemented yet!')
            elif cluster_evaluation_method == 'dunns_index':
                # cluster_eval_score =
                raise NotImplemented(f'The {cluster_evaluation_method} clustering evaluation method is not '
                                     f'implemented yet!')

            clusters_metric_scores[label] = cluster_eval_score

        mean_clustering_score = np.array(list(clusters_metric_scores.values())).mean()

        return clusters_metric_scores, mean_clustering_score
