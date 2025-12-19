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
sys.path.append(os.sep.join(os.getcwd().split(os.sep)[:-1]))
from src.utils import *
from src.quanta_utils import *
from src.SegregationIdx import SegregationIdx
from src.uSpiCalc import uSpiCalc
import argparse


def get_argparser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "-c",
        "--config",
        type=str,
        default="/Users/esraan/CodeBase/CellDeathSpreading/configs/configs.yml",
        help="Config file path (other given arguments will superseed this).",
    )
    p.add_argument("--data_dir", type=str, default="/Users/esraan/CodeBase/CellDeathSpreading/data/", help="Directory for data.")
    p.add_argument("--sliding_window_size",
           type=int, 
           nargs="+",
           default=[5], 
           help="Size of the sliding window.")
    p.add_argument("--distance_threshold",
           type=int, 
           nargs="+",
           default=[100], 
           help="Distance threshold values.")
    p.add_argument("--n_permutations", type=int, default=1000, help="Number of permutations.")
    p.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--results_dir", type=str, default="/Users/esraan/CodeBase/CellDeathSpreading/results/", help="Directory for results.")
    p.add_argument("--spi_csv_file", type=str, default="AllExperimentsSPIs.csv", help="CSV file name for SPI results.")
    p.add_argument("--ssi_csv_file", type=str, default="SSI_sensitivity.csv", help="CSV file name for SSI results.")
    p.add_argument("--files_to_analyze", 
          type=str, 
          nargs="+", 
          default=["mixed"], 
          help="List of files to analyze.")
    p.add_argument("--death_annotations_dir",
           type=str,
           default="mixed_death_annotations/",
           help="Directory for death annotations.")
    p.add_argument("--metadata_file",
           type=str,
           default="metadata.csv",
           help="Metadata file name of the relevant csvs.")
    p.add_argument("--save_csv", type=bool, default=True, help="Whether to save the results as a CSV file.")
    p.add_argument("--run_analysis", type=str, default="ssi", help="Which analysis to run: 'ssi', 'spi', or 'all'.")

    return p

if __name__ == "__main__":
    args = get_argparser().parse_args()

    data_dir = args.data_dir
    sliding_window_size = args.sliding_window_size
    distance_threshold = args.distance_threshold
    n_permutations = args.n_permutations
    random_seed = args.random_seed
    results_dir = args.results_dir
    death_annotations_dir = os.path.join(data_dir,args.death_annotations_dir)
    metadata_file = args.metadata_file
    np.random.seed(random_seed) 
    # Create results directory if it doesn't exist
    final_dataframe = pd.DataFrame()
    os.makedirs(results_dir, exist_ok=True)
    metadata = pd.read_csv(os.path.join(data_dir, metadata_file))
    # Filter metadata based on files to analyze
    files_to_analyze = args.files_to_analyze
    metadata = metadata[metadata["File Name"].str.contains('|'.join(files_to_analyze), case=False, na=False)]
    if metadata.empty:
        raise ValueError("No metadata found for the specified files to analyze.")
    if args.run_analysis in ["spi", "all"]:
        for neighbors_dist_threshold in distance_threshold:
            for sliding_time_window_size in sliding_window_size:
                p_nucs_by_exp_name = {}
                global_density_by_exp_name = {}
                all_experiments_spi_regeneration = calc_all_experiments_SPI_for_figure(
                    exp_name=list(metadata["File Name"]),
                    exps_dir_path=death_annotations_dir,
                    meta_data_full_file_path=os.path.join(data_dir, metadata_file),
                    dist_threshold=neighbors_dist_threshold,
                    time_unit="minutes",
                    sliding_time_window_size=sliding_time_window_size,
                    n_scramble=n_permutations
                )
                reformatting_all_previos_experiments_spi_and_ni_regeneration = {"Experiment_name": [],
                                                                                "SPI": [],
                                                                                "Treatment": [],
                                                                                "Cell Line + Treatment": [],
                                                                                "Cell Line": [],
                                                                                "Origin": [],
                                                                                "Mode": [],
                                                                                "Density": [],
                                                                                "pvalue": [],
                                                                                "sliding_time_window_size": [],
                                                                                "neighbors_dist_threshold": []}
                for key, value in all_experiments_spi_regeneration.items():
                    if value is None:
                        continue
                    origin = metadata[metadata["File Name"] == key]["Origin"].values[0]
                    mode = metadata[metadata["File Name"] == key]["Mode"].values[0]
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["Origin"].append(origin)
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["Mode"].append(mode)
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["Experiment_name"].append(key)
                    exp_cell_line = metadata[metadata["File Name"] == key]["Cell Line"].values[0]
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["SPI"].append(value[0])
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["pvalue"].append(value[1])
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["Cell Line"].append(exp_cell_line)
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["Cell Line + Treatment"].append(replace_ugly_long_name(key, exp_cell_line))
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["Treatment"].append(simple_treatment(key))
                    density = metadata[metadata["File Name"] == key]["Density(#Cells)"].values[0]
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["Density"].append(density)
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["sliding_time_window_size"].append(sliding_time_window_size)
                    reformatting_all_previos_experiments_spi_and_ni_regeneration["neighbors_dist_threshold"].append(neighbors_dist_threshold)
                
                reformatting_all_previos_experiments_spi_and_ni_regeneration_df = pd.DataFrame(reformatting_all_previos_experiments_spi_and_ni_regeneration)

                final_dataframe = pd.concat([final_dataframe, reformatting_all_previos_experiments_spi_and_ni_regeneration_df], ignore_index=True)
        if args.save_csv:
            final_dataframe.to_csv(results_dir + args.spi_csv_file, index=False)
    
    if args.run_analysis in ["ssi", "all"]:

        for dist in distance_threshold:
            results_by_distance = {}
            res = {}
            for file in list(metadata["File Name"]):
                if "_mixed" not in file:
                    continue
                exp_full_path = os.path.join(death_annotations_dir, file)
                if not os.path.exists(exp_full_path):
                    continue  # skip if file is missing
                cells_loci, cells_times_of_death = read_experiment_cell_xy_and_death_times(
                    exp_full_path=exp_full_path
                )
                csv_file = pd.read_csv(exp_full_path)
                death_modes = csv_file[["Mode"]].values
                segregation_index = SegregationIdx(
                    cells_loci,
                    death_modes,
                    args.n_permutations,
                    dist,
                    filter_neighbors_by_distance=True,
                    neighbors_level=1,
                    stats_to_calculate="mean",
                )
                seg_res_dict = segregation_index.get_segregation_index()

                # store (SSI, pvalue) only for apoptosis / necrosis
                res[file] = {
                    key: (value[0] / value[2], value[1])
                    for key, value in seg_res_dict.items()
                    if key in ("apoptosis", "necrosis")
                }

            # map original file names to short labels when available
            # plot_res_dict = {new_res.get(key, key): value for key, value in res.items()}
            results_by_distance[dist] = res

        # ---- reshape results_by_distance into a flat DataFrame and save as CSV ----
        rows = []

        for dist, files_dict in results_by_distance.items():
            for short_name, modes_dict in files_dict.items():
                for mode, (ssi, pval) in modes_dict.items():
                    rows.append(
                        {
                            "File Name": short_name, 
                            "Origin":metadata[metadata["File Name"]==short_name]["Origin"].values[0],
                            "neighbors_dist_threshold": dist,
                            "Mode": mode,
                            "SSI": ssi,
                            "pvalue": pval,
                        }
                    )

        ssi_df = pd.DataFrame(rows)
        if args.save_csv:
            out_path = results_dir + args.ssi_csv_file
            ssi_df.to_csv(out_path, index=False)
    