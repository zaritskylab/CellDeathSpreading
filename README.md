# Ferroptosis induces heterogeneous death profiles that are controlled by lysosome rupture
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281-blue)](https://doi.org/10.5281/zenodo.18271042)

Jyotirekha Das<sup>1*</sup>, Saloni K. Hombalkar<sup>1,2*</sup>, Alison D. Klein<sup>1,2</sup>, Esraa Nsasra<sup>3</sup>, Muskaan Vasandani<sup>1,4</sup>, Kay Petruzzi<sup>1</sup>, Dajun Lu<sup>1</sup>, Stephen Ruiz<sup>2,5</sup>, Orit Kliper-Gross<sup>3</sup>, Jiachen Hu<sup>1,4</sup>, Michelle Riegman<sup>1,4</sup>, Xuejun Jiang<sup>1</sup>, Daniel A. Heller<sup>5</sup>, Assaf Zaritsky<sup>3</sup>, Michelle S. Bradbury<sup>6</sup>, and Michael Overholtzer<sup>1,2,4</sup>
  
  *Equal contribution
1. Cell Biology Program, Memorial Sloan Kettering Cancer Center, New York, NY 10065
2. BCMB Graduate Program, Weill Cornell Medical College, New York, NY 10065
3. Institute for Interdisciplinary Computational Science, Faculty of Computer and Information Science, Ben-Gurion University of the Negev, Beer-Sheva 84105, Israel
4. Gerstner Sloan Kettering Graduate School of Biomedical Sciences, Memorial Sloan Kettering Cancer Center, New York, NY 10065
5. Molecular Pharmacology Program, Memorial Sloan Kettering Cancer Center, New York, New York 10065
6. Molecular Imaging Innovations Institute, Department of Radiology, Weill Cornell Medical College, New York, NY 10065



## Table of Contents
1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Reproducing Paper Results](#6-reproducing-paper-results)
7. [Citation](#7-citation)
8. [License](#8-license)
9. [Acknowledgments](#9-acknowledgments)
10. [Contact](#10-contact)
---

## 1. Overview
A computational framework for **quantifying collective ferroptosis dynamics from live-cell imaging**. Using **single-cell death times**, **morphological death fates** (necrotic vs apoptotic-like), and **cell positions**, the framework computes spatiotemporal statistics that distinguish **single-cell** from **propagative** ferroptosis:

- **Spatial Propagation Index (SPI):** quantifies whether neighboring cells of the same fate die **more synchronously** than expected by chance.
- **Spatial Segregation Index (SSI):** quantifies whether different death fates cluster in space.

These metrics enable systematic comparison across perturbations (e.g., **GPX4 inhibition** vs **glutathione depletion**) and help reveal mechanisms underlying **locally propagative, lysosome-linked death waves** in cell populations.

To examine the full paper, please visit [URL]

![Overview figure](figures/overview.png)


---

## 2. Key Features

- **Cell death quantification** of collective ferroptosis from time-lapse imaging.
- **Spatial segregation index** evaluating tendency of same death group to cluster.
- **Voronoi-based neighboring cells detection** with a biologically motivated distance cutoff of 100 microns.
- **Permutation testing (bootstrapping)** for statistical significance of SSI and SPI.
- Reproducible scripts for **preprocessing → SSI/SPI computation → plotting and figure generation**
- Designed to support **multiple biological replicates** and varying acquisition rates (e.g., 5-min and 10-min imaging)

---

## 3. Repository Structure
```
CellDeathSpreading/
├── README.md
├── environment.yml                   # full requirments for installing the relevant enviroment 
├── main.py
├── LICENSE
├── figures/                          # Publication-quality figures
├── notebooks/                        # notebooks
│   ├── paper_figures/                # generating the paper figure of SPI and SSI analysis
│   └── statistical_signifigance/     # performing permutation tests + p-values
│             
├── src/                              # main scripts of SSI and SPI quantifications
├── data/   
│   ├── death_annotations             # folder of csvs: manual annotation of death times and modes
│   └── fig2gh_metadata.csv           # fig2gh_metadata of manually annotated csvs in death_annotations/  
│ 
└── results/                          # SSI and SPI analysis results will stored in here    
```
---

## 4. Installation

### Quick Start

```bash
# 1) Clone and enter repo
git clone https://github.com/zaritskylab/CellDeathSpreading.git
cd CellDeathSpreading

# 2) Create and activate environment
conda env create -f environment.yml
conda activate CellDS
```

## 5. Usage

Experiments are configured via command‑line arguments


### 5.1 Example 1: SPI Analysis Only

```
# Run SPI only, over all samples - results will be saved in results in a csv file "SPI_calculations.csv" unless configured differently
python main.py \
  --run_analysis spi \
  --data_dir data/death_annotations \   #path to data
  --results_dir results \               #path to where results intended to be saved
  --sliding_window_size 10 \            #time resolution to be used
  --distance_threshold 100 \            # distance threshold of neighboring cells
  --n_permutations 1000 
```
### 5.2 Example 2: SSI Analysis Only

```
# Running SSI only over all samples - results will be saved in results in a csv file "SSI_calculations.csv" unless configured differently
python main.py \
  --run_analysis ssi \
  --data_dir data/death_annotations \
  --results_dir results \
  --distance_threshold 100
```
### 5.3 Example 3: Combined SPI and SSI Analysis

```
# Run both SPI and SSI and save using custom CSV names 
python main.py \
  --run_analysis all \
  --spi_csv_file SPI_calculations.csv \
  --ssi_csv_file SSI_calculations.csv
```
### 5.4 Running the SSI and SPI Analysis on New Data

Input data should be CSV files with the following columns:

      death_time: Time of cell death
      death_mode: Morphological death fate (e.g., apoptosis vs necrosis)
      cell_x: X-coordinate of cell position
      cell_y: Y-coordinate of cell position


fig2gh_metadata csv should be provided with the same template in data/ dir based on fig2gh_metadata extracted from the raw time-lapse. fig2gh_metadata must contain the following columns: File Name, Treatment, Cell Line, SizeX, SizeY, PhysicalResolution (um/px), Time Interval (min), Region. 

---

## 6. Reproducing Paper Results
To regenerate all figures from the paper:

1. **Run the complete analysis:**
   ```bash
   python main.py --run_analysis all
   ```
This will run the analysis by paper parameters. Results will be saved in `results/` as `SPI_calculations.csv` and `SSI_calculations.csv`.

2. **Generate figures:**
   Open and execute `notebooks/paper_figures.ipynb` sequentially.

---

---

## 7. Citation

If you use this code in academic work, please cite the associated paper.

**Journal version (recommended):**

```bibtex
@article{das_inpress_ferroptosis,
  title   = {Ferroptosis induces heterogeneous death profiles that are controlled by lysosome rupture},
  author  = {Das, Jyotirekha and Hombalkar, Saloni K. and Klein, Alison D. and Nsasra, Esraa and Vasandani, Muskaan and Petruzzi, Kay and Lu, Dajun and Ruiz, Stephen and Kliper-Gross, Orit and Hu, Jiachen and Riegman, Michelle and Jiang, Xuejun and Heller, Daniel A. and Zaritsky, Assaf and Bradbury, Michelle S. and Overholtzer, Michael},
  journal = {Developmental Cell},
  year    = {2025},
  note    = {In press},
}

```

---

## 8. License

This repository (including data, documentation, and figures where applicable) is intended for academic/research use and is released under **CC BY-NC 4.0**. See [LICENSE](LICENSE) for details.

---


## 9. Acknowledgments


##### This repository builds upon [CellDeathQuantification](https://github.com/Yishaiaz/CellDeathQuantification).
---

## 10. Contact

For questions, issues, or collaboration inquiries:
- **Esraa Nsasra**: esraan@post.bgu.ac.il/esraansas@gmail.com
- **Assaf Zaritsky**: assafzar@gmail.com

Please open an [issue](https://github.com/zaritskylab/CellDeathSpreading/issues) for bug reports or feature requests.
