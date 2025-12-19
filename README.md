# CellDeathSpreading

A computational framework for **quantifying collective ferroptosis dynamics from live-cell imaging**. Using **single-cell death times**, **morphological death fates** (necrotic vs apoptotic-like), and **cell positions**, the framework computes spatiotemporal statistics that distinguish **single-cell** from **propagative** ferroptosis:

- **Spatial Segregation Index (SSI):** quantifies whether different death fates cluster in space.
- **Spatial Propagation Index (SPI):** quantifies whether neighboring cells of the same fate die **more synchronously** than expected by chance.

These metrics enable systematic comparison across perturbations (e.g., **GPX4 inhibition** vs **glutathione depletion**) and help reveal mechanisms underlying **locally propagative, lysosome-linked death waves** in cell populations.

To examine the full paper, please visit [URL]

![Overview figure](figures/overview.png)


---

## Key Features

- **Cell death quantification** of collective ferroptosis from time-lapse imaging.
- **Spatial segregation index** evaluating tendency of same death group to cluster.
- **Voronoi-based neighboring cells detection** with a biologically motivated distance cutoff of 100 microns.
- **Permutation testing (bootstrapping)** for statistical significance of SSI and SPI.
- Reproducible scripts for **preprocessing → SSI/SPI computation → plotting and figure generation**
- Designed to support **multiple biological replicates** and varying acquisition rates (e.g., 5-min and 10-min imaging)

---

## Repository Structure
```
CellDeathSpreading/
├── README.md
├── environment.yml             # full requirments for installing the relavant enviroment to generate the results in the paper and apply it on other datab
├── main.py
├── LICENSE
├── figures/                    # figures in paper that were the results of this analysis
├── notebooks/                  # notebooks
│   ├── paper_figures/                # generating the paper figure of SPI and SSI analysis
│   ├── sensitivity_analysis/                  # performing sensitivity analysis of distance threshold and time sliding window 
│   └── statistical_signifigance/  
│             # reproming permutation tests + p-values
├── src/                        # main scripts of SSI and SPI quantifications
├── data/   
│   ├──  mixed_death_annotations  # csvs of manual annotation of death times and modes
│   └── metadata.csv                # metadata of expreiments that were manually annotated in mixed_death_annotations  
│ 
└── results/                    # csvs summery of running the analysis on data stored in /data    
```
---

## Setup Instructions

### Quick Start

```bash
# 1) Clone and enter repo
git clone https://github.com/zaritskylab/CellDeathSpreading.git
cd CellDeathSpreading

# 2) Create and activate environment
conda env create -f environment.yml
conda activate CellDS
```

### Configuration

Experiments are configured via command‑line arguments


Run Example 1
```
# Run SPI only, using mixed experiments
python main.py \
  --run_analysis spi \
  --data_dir /path/to/data \
  --results_dir /path/to/results \
  --files_to_analyze mixed \
  --sliding_window_size 5 10 \
  --distance_threshold 50 100 \
  --n_permutations 1000
```
Run Example 2
```
# Run SSI only over mixed co-culture experiments
python main.py \
  --run_analysis ssi \
  --data_dir /path/to/data \
  --results_dir /path/to/results \
  --files_to_analyze mixed \
  --distance_threshold 100
```
Run Example 3
```
# Run both SPI and SSI and save to custom CSV names
python main.py \
  --run_analysis all \
  --spi_csv_file AllExperimentsSPIs_updated.csv \
  --ssi_csv_file SSI_sensitivity.csv
```
## To run main on new data:
The data should be csvs of time of death, mode of death and the location of cells with columns: death_time, death_mode, cell_x, cell_y.

Metadata csv should be provided with the same template in /data dir based on metadata extracted from the raw time-lapse.
## Regentation of paper figures:
Regenration of paper figures can be performed in notebooks/paper_figures.ipynb

---

## Citation

If you use this code in academic work, please cite the associated paper.

**Journal version (recommended):**

```bibtex
@article{das_inpress_ferroptosis,
  title   = {Ferroptosis induces heterogeneous death profiles that are controlled by lysosome rupture},
  author  = {Das, Jyotirekha and Hombalkar, Saloni K. and Klein, Alison D. and Nsasra, Esraa and Vasandani, Muskaan and Petruzzi, Kay and Lu, Dajun and Ruiz, Stephen and Kliper-Gross, Orit and Hu, Jiachen and Riegman, Michelle and Jiang, Xuejun and Heller, Daniel A. and Zaritsky, Assaf and Bradbury, Michelle S. and Overholtzer, Michael},
  journal = {Developmental Cell},
  year    = {2025},
  note    = {In press},
  doi     = {TBD}
}

```

---

## License

This repository (including data, documentation, and figures where applicable) is intended for academic/research use and is released under **CC BY-NC 4.0**. See [LICENSE](LICENSE) for details.


---

## Contact

Please contact esraan@post.bgu.ac.il or assafzar@gmail.com for comments or questions regarding this repo.