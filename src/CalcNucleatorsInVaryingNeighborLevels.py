import os
from typing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from NucleatorsProbabilities import *
from src.utils import *
from global_parameters import *
from Visualization import *
from scipy.stats import wilcoxon, kruskal, mannwhitneyu


def calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths: np.array,
                                                         cells_neighbors_lists: List[List]):
    neighbors_of_dead_cells_probability_to_die_at_all_frames = []
    implicit_temporal_resolution = cells_times_of_deaths[1] - cells_times_of_deaths[0]
    unique_times_of_death = np.unique(cells_times_of_deaths)
    for timeframe in unique_times_of_death:
        # get all dead cells at time = timeframe
        dead_cells_at_timeframe = np.where(cells_times_of_deaths == timeframe)[0]
        alive_cells_at_timeframe = np.where(cells_times_of_deaths > timeframe)[0]
        # get all dead cells at time = timeframe + implicit_temporal_resolution
        dead_cells_at_next_timeframe = np.where(cells_times_of_deaths == (timeframe + implicit_temporal_resolution))[0]
        # get all neighbors of dead_cells_at_timeframe
        neighbors_of_dead_cells = set()
        for dead_cell_idx in dead_cells_at_timeframe:
            neighbors_of_dead_cells.update(cells_neighbors_lists[dead_cell_idx])
        # keep only alive cells as neighbors at timeframe
        neighbors_of_dead_cells = neighbors_of_dead_cells.intersection(
            set(alive_cells_at_timeframe))

        # get all cells in neighbors_of_dead_cell that are dead in next timeframe and were alive before
        dead_neighbors_at_next_timeframe_of_dead_cells = neighbors_of_dead_cells.copy()
        # intersect with dead cells at timeframe_ + implicit_temporal_resolution
        dead_neighbors_at_next_timeframe_of_dead_cells = dead_neighbors_at_next_timeframe_of_dead_cells.intersection(
            set(dead_cells_at_next_timeframe))
        if len(neighbors_of_dead_cells) == 0 and len(dead_neighbors_at_next_timeframe_of_dead_cells) != 0:
            raise RuntimeWarning(
                f'number of alive neighbors of dead cells is zero, but number of dead neighbors of dead cells is not! timeframe:{timeframe}')
        neighbors_of_dead_cells_probability_to_die_at_curr_frame = len(
            dead_neighbors_at_next_timeframe_of_dead_cells) / (len(neighbors_of_dead_cells) + EPSILON)
        neighbors_of_dead_cells_probability_to_die_at_all_frames.append(
            neighbors_of_dead_cells_probability_to_die_at_curr_frame)
    return neighbors_of_dead_cells_probability_to_die_at_all_frames


def calc_pnuc_at_varying_distances_of_neighbors_single_exp(exp_filename,
                                                           exp_main_directory_path,
                                                           file_details_full_path,
                                                           path_to_save_fig_no_type='',
                                                           **kwargs) -> Tuple[np.array, np.array, np.array]:
    """
    plots p(nuc) at two different distances from dead cells for a single experiment
    :param exp_filename: the experiment XYT file name (including .csv)
    :param exp_main_directory_path: the directory in which the file resides.
    :param file_details_full_path: the full path to the file details csv file.
    :param func_mode: either single or multi. multi returns the signal instead of plotting it.
    :param plot_kwargs:
    :return: Tuple of np.array = (nuc_probas_calculator_lvl_1, nuc_probas_calculator_lvl_2, nuc_probas_calculator_lvl_3)
    """
    visualize_single_exp = kwargs.get('visualize_single_exp', True)

    exp_path = os.sep.join([exp_main_directory_path, exp_filename])
    exp_xyt = pd.read_csv(exp_path)
    exp_details_df = pd.read_csv(file_details_full_path)
    full_x = exp_xyt["cell_x"].values
    full_y = exp_xyt["cell_y"].values
    # n_instances = len(full_x)
    die_times = exp_xyt["death_time"].values
    XY = np.column_stack((full_x, full_y))
    exp_temporal_resolution = exp_details_df[exp_details_df['File Name'] == exp_filename]['Time Interval (min)'].values[
        0]
    exp_treatment_type = exp_details_df[exp_details_df['File Name'] == exp_filename]['Treatment'].values[0]
    jump_interval = exp_temporal_resolution
    # time_window_size = WINDOW_SIZE * jump_interval
    cmap = mpl.cm.__builtin_cmaps[13]

    # get neighbors list of all cells (topological by Voronoi)
    neighbors_list, neighbors_list2, neighbors_list3 = get_cells_neighbors(XY=XY,
                                                                           threshold_dist=DIST_THRESHOLD_IN_PIXELS)

    nuc_probas_calculator_lvl_1 = np.asarray(
        calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths=die_times,
                                                             cells_neighbors_lists=neighbors_list))

    nuc_probas_calculator_lvl_2 = np.asarray(
        calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths=die_times,
                                                             cells_neighbors_lists=neighbors_list2))
    nuc_probas_calculator_lvl_3 = np.asarray(
        calc_probability_to_die_as_a_neighbor_at_given_level(cells_times_of_deaths=die_times,
                                                             cells_neighbors_lists=neighbors_list3))
    org_path = path_to_save_fig_no_type
    path_to_save_fig_no_type = org_path + '_2on3'

    if visualize_single_exp:
        # get visualization argument
        probabilities_marker_size = kwargs.get('probabilities_marker_size', 300)

        scatter_with_linear_regression_line(nuc_probas_calculator_lvl_2, nuc_probas_calculator_lvl_3,
                                            x_label='Nucleation probability at level 2 neighborhood',
                                            y_label='Nucleation probability at level 3 neighborhood',
                                            title=f'experiment treatment: {exp_treatment_type}',
                                            path_to_save_fig=path_to_save_fig_no_type,
                                            colors=np.unique(die_times),
                                            color_map=cmap,
                                            marker_size=probabilities_marker_size)

        path_to_save_fig_no_type = org_path + '_1on2'
        scatter_with_linear_regression_line(nuc_probas_calculator_lvl_1, nuc_probas_calculator_lvl_2,
                                            x_label='Nucleation probability at level 1 neighborhood',
                                            y_label='Nucleation probability at level 2 neighborhood',
                                            title=f'experiment treatment: {exp_treatment_type}',
                                            path_to_save_fig=path_to_save_fig_no_type,
                                            colors=np.unique(die_times),
                                            color_map=cmap,
                                            marker_size=probabilities_marker_size)

        path_to_save_fig_no_type = org_path + '_1on3'
        scatter_with_linear_regression_line(nuc_probas_calculator_lvl_1, nuc_probas_calculator_lvl_3,
                                            x_label='Nucleation probability at level 1 neighborhood',
                                            y_label='Nucleation probability at level 3 neighborhood',
                                            title=f'experiment treatment: {exp_treatment_type}',
                                            path_to_save_fig=path_to_save_fig_no_type,
                                            colors=np.unique(die_times),
                                            color_map=cmap,
                                            marker_size=probabilities_marker_size)

    return nuc_probas_calculator_lvl_1, nuc_probas_calculator_lvl_2, nuc_probas_calculator_lvl_3


def calc_pnuc_at_varying_distances_of_neighbors_multiple_exps(main_exp_dir_full_path: str,
                                                              limit_exp_num: int = float('inf'),
                                                              **kwargs):
    """
    plots p(nuc) at two different distances from dead cells for multiple experiments.
    Each experiments mean p(nuc) at a given distance is calculated, and the
    Wilcoxon rank measurement is calculated on the distance of each datapoint (a single
    experiment mean) from the x=y regression line.
    :param main_exp_dir_full_path:
    :param limit_exp_num:
    :param kwargs:
    :return:
    """

    treatments_to_include = kwargs.get('treatments_to_include', 'all')
    temporal_resolutions_to_include = kwargs.get('temporal_resolutions_to_include', [30])
    kwargs['visualize_single_exp'] = False
    plot_mean_of_means = kwargs.get('plot_mean_of_means', False)
    calc_erastin_and_cdots_errors = kwargs.get('calc_erastin_and_cdots_errors', False)
    calc_wilcoxon_rank = kwargs.get('calc_wilcoxon_rank', True) * calc_erastin_and_cdots_errors

    all_files_to_analyze_full_paths, all_files_to_analyze_only_exp_names = \
        get_all_paths_csv_files_in_dir(dir_path=main_exp_dir_full_path)

    meta_data_file_path = os.sep.join(main_exp_dir_full_path.split(os.sep)[:-1] + ['ExperimentsMetaData.csv'])

    all_exps_mean_p_nuc_lvl1, all_exps_mean_p_nuc_lvl2, all_exps_mean_p_nuc_lvl3, all_exps_results_by_treatments = \
        list(), list(), list(), dict()
    all_exps_treatments = list()

    total_exps = len(all_files_to_analyze_only_exp_names)

    for exp_idx, exp_details in enumerate(zip(all_files_to_analyze_full_paths, all_files_to_analyze_only_exp_names)):
        if limit_exp_num < exp_idx + 1:
            break

        file_full_path, exp_name = exp_details

        exp_treatment, exp_temporal_res = get_exp_treatment_type_and_temporal_resolution(
            exp_file_name=exp_name + '.csv', meta_data_file_full_path=meta_data_file_path,
            compressed_flag=False)

        print(f'analyzing exp {exp_name} | {exp_idx + 1}/{total_exps}')

        # skip un-wanted treatments
        if treatments_to_include != 'all' and \
                verify_any_str_from_lst_in_specific_str(exp_treatment, treatments_to_include) \
                or int(exp_temporal_res) not in temporal_resolutions_to_include:
            continue

        exps_mean_p_nuc_lvl1, \
        exps_mean_p_nuc_lvl2, \
        exps_mean_p_nuc_lvl3 = calc_pnuc_at_varying_distances_of_neighbors_single_exp(exp_name + '.csv',
                                                                                      main_exp_dir_full_path,
                                                                                      meta_data_file_path,
                                                                                      path_to_save_fig_no_type='',
                                                                                      **kwargs)
        all_exps_mean_p_nuc_lvl1.append(np.mean(exps_mean_p_nuc_lvl1))
        all_exps_mean_p_nuc_lvl2.append(np.mean(exps_mean_p_nuc_lvl2))
        all_exps_mean_p_nuc_lvl3.append(np.mean(exps_mean_p_nuc_lvl3))
        all_exps_treatments.append(exp_treatment)

        treatment_res_dict = all_exps_results_by_treatments.get(exp_treatment, {})
        treatment_res_dict['lvl1'] = treatment_res_dict.get('lvl1', []) + [np.mean(exps_mean_p_nuc_lvl1)]
        treatment_res_dict['lvl2'] = treatment_res_dict.get('lvl2', []) + [np.mean(exps_mean_p_nuc_lvl2)]
        treatment_res_dict['lvl3'] = treatment_res_dict.get('lvl3', []) + [np.mean(exps_mean_p_nuc_lvl3)]

        all_exps_results_by_treatments[exp_treatment] = treatment_res_dict

    if plot_mean_of_means:
        all_exps_mean_p_nuc_lvl2 = list()
        all_exps_mean_p_nuc_lvl3 = list()
        all_exps_mean_p_nuc_lvl1 = list()
        for treatment_name, treatment_results in all_exps_results_by_treatments.items():
            all_exps_mean_p_nuc_lvl1.append(np.array(treatment_results['lvl1']).mean())
            all_exps_mean_p_nuc_lvl2.append(np.array(treatment_results['lvl2']).mean())
            all_exps_mean_p_nuc_lvl3.append(np.array(treatment_results['lvl3']).mean())

        all_exps_treatments = list(all_exps_results_by_treatments.keys())

    if calc_wilcoxon_rank:
        if plot_mean_of_means:
            raise AttributeError('Can not calculate Wilcoxon rank on mean of means (single datapoint per experiment)')
        treatments_errors = {}
        for treatment_name, treatment_values in all_exps_results_by_treatments.items():
            treatment_lvl1 = treatment_values['lvl1']
            treatment_lvl2 = treatment_values['lvl2']
            treatment_lvl3 = treatment_values['lvl3']

            regression_line = np.linspace(0, 1, num=len(treatment_lvl1), endpoint=True)

            lvl1_vs_lvl2_errors = np.sqrt(
                (regression_line - treatment_lvl1) ** 2 + (regression_line - treatment_lvl2) ** 2)
            lvl1_vs_lvl3_errors = np.sqrt(
                (regression_line - treatment_lvl1) ** 2 + (regression_line - treatment_lvl3) ** 2)
            lvl2_vs_lvl3_errors = np.sqrt(
                (regression_line - treatment_lvl2) ** 2 + (regression_line - treatment_lvl3) ** 2)

            treatments_errors[treatment_name] = {'lvl1_vs_lvl2_errors': lvl1_vs_lvl2_errors,
                                                 'lvl1_vs_lvl3_errors': lvl1_vs_lvl3_errors,
                                                 'lvl2_vs_lvl3_errors': lvl2_vs_lvl3_errors}
        if calc_erastin_and_cdots_errors:
            treatments_errors_no_erastin = {key: val for key, val in treatments_errors.items()
                                            if 'erastin' not in key.lower()}
            treatments_errors_no_cdots = {key: val for key, val in treatments_errors.items()
                                          if "C' dots" not in key.lower()}
            erastin_errors = list({key: val for key, val in treatments_errors.items()
                              if 'erastin' in key.lower()}.values())[0]
            cdots_errors = list({key: val for key, val in treatments_errors.items()
                            if "C' dots" not in key.lower()}.values())[0]

            # calculating_erastin_scores
            erastin_statistics_and_pvals_vs_all_other_treatments = {}
            for treatment_name, treatment_error_scores in treatments_errors_no_erastin.items():
                statistic, p_val = mannwhitneyu(erastin_errors['lvl1_vs_lvl2_errors'],
                                            treatment_error_scores['lvl1_vs_lvl2_errors'])
                erastin_statistics_and_pvals_vs_all_other_treatments[treatment_name] = {
                    'statistic': statistic,
                    'p_val': p_val
                }

            print('#' * 20)
            print('Wilcoxon scores on errors: Erastin Vs. all other treatments:')
            print(erastin_statistics_and_pvals_vs_all_other_treatments)
            print('#' * 20)
        if calc_wilcoxon_rank:
            cdots_statistics_and_pvals_vs_all_other_treatments = {}
            for treatment_name, treatment_error_scores in treatments_errors_no_cdots.items():
                statistic, p_val = mannwhitneyu(cdots_errors['lvl1_vs_lvl2_errors'],
                                                treatment_error_scores['lvl1_vs_lvl2_errors'])
                cdots_statistics_and_pvals_vs_all_other_treatments[treatment_name] = {
                    'statistic': statistic,
                    'p_val': p_val
                }

            print('#' * 20)
            print("Wilcoxon scores on errors: C'Dots Vs. all other treatments:")
            print(cdots_statistics_and_pvals_vs_all_other_treatments)
            print('#' * 20)

    # get visualization argument
    probabilities_marker_size = kwargs.get('probabilities_marker_size', 100)

    scatter_with_linear_regression_line_about_treatment(all_exps_mean_p_nuc_lvl2, all_exps_mean_p_nuc_lvl3,
                                                        x_label='Nucleation probability at level 2 neighborhood',
                                                        y_label='Nucleation probability at level 3 neighborhood',
                                                        title=f'Treatments µ(P(Nuc)) at Various neighborhood levels',
                                                        marker_size=probabilities_marker_size,
                                                        exps_treatments=all_exps_treatments)

    scatter_with_linear_regression_line_about_treatment(all_exps_mean_p_nuc_lvl1, all_exps_mean_p_nuc_lvl2,
                                                        x_label='Nucleation probability at level 1 neighborhood',
                                                        y_label='Nucleation probability at level 2 neighborhood',
                                                        title=f'Treatments µ(P(Nuc)) at Various neighborhood levels',
                                                        marker_size=probabilities_marker_size,
                                                        exps_treatments=all_exps_treatments)

    scatter_with_linear_regression_line_about_treatment(all_exps_mean_p_nuc_lvl1, all_exps_mean_p_nuc_lvl3,
                                                        x_label='Nucleation probability at level 1 neighborhood',
                                                        y_label='Nucleation probability at level 3 neighborhood',
                                                        title=f'Treatments µ(P(Nuc)) at Various neighborhood levels',
                                                        marker_size=probabilities_marker_size,
                                                        exps_treatments=all_exps_treatments)


if __name__ == '__main__':
    # MULTIPLE EXPERIMENT TESTING
    # # each experiment plotted separately
    # exp_main_dir_path = '../Data/Experiments_XYT_CSV/OriginalTimeMinutesData'
    # exp_meta_data_full_path = '../Data/Experiments_XYT_CSV/ExperimentsMetaData.csv'
    # for file_idx, filename in enumerate(filter(lambda x: x.endswith('.csv'), os.listdir(exp_main_dir_path))):
    #     print(f'file :{filename}|{file_idx + 1}/'
    #           f'{len(list(filter(lambda x: x.endswith(".csv"), os.listdir(exp_main_dir_path))))}')
    #     path_to_save_fig = os.sep.join(
    #         ['../Results', 'NucleationProbabilitiesForVaryingLevelsOfNeighborhoods', filename.replace('.csv', '')])
    #     calc_pnuc_at_varying_distances_of_neighbors_single_exp(exp_filename=filename,
    #                                                            exp_main_directory_path=exp_main_dir_path,
    #                                                            file_details_full_path=exp_meta_data_full_path,
    #                                                            path_to_save_fig_no_type=path_to_save_fig)
    # all experiments plotted by treatment

    main_exp_dir_full_path = 'C:\\Users\\User\\PycharmProjects\\CellDeathQuantification\\Data\\Experiments_XYT_CSV\\OriginalTimeMinutesData'
    kwargs = {'plot_mean_of_means': False}
    calc_pnuc_at_varying_distances_of_neighbors_multiple_exps(main_exp_dir_full_path,
                                                              limit_exp_num=float('inf'),
                                                              treatments_to_include=['TNFa'],
                                                              temporal_resolutions_to_include=[60],
                                                              **kwargs)

    # SINGLE EXPERIMENT TESTING
    # calc_pnuc_at_varying_distances_of_neighbors(exp_filename='20180620_HAP1_erastin_xy7.csv',
    #                                             exp_main_directory_path=exp_main_dir_path,
    #                                             file_details_full_path=exp_meta_data_full_path)
    # calc_pnuc_at_varying_distances_of_neighbors(exp_filename='20181227_MCF10A_SKT_xy4.csv',
    #                                             exp_main_directory_path=exp_main_dir_path,
    #                                             file_details_full_path=exp_meta_data_full_path)
