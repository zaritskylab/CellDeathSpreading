# CellDeathSpreading

A computational framework for **quantifying collective ferroptosis dynamics from live-cell imaging**. Using **single-cell death times**, **morphological death fates** (necrotic vs apoptotic-like), and **cell positions**, the framework computes spatiotemporal statistics that distinguish **single-cell** from **propagative** ferroptosis:

- **Spatial Segregation Index (SSI):** quantifies whether different death fates cluster in space.
- **Spatial Propagation Index (SPI):** quantifies whether neighboring cells of the same fate die **more synchronously** than expected by chance.

These metrics enable systematic comparison across perturbations (e.g., **GPX4 inhibition** vs **glutathione depletion**) and help reveal mechanisms underlying **locally propagative, lysosome-linked death waves** in cell populations.

![Overview figure](figures/overview.png)


---

## Key Features

- **Cell death quantification** of collective ferroptosis from time-lapse imaging
- **Spatial segregation index** (necrotic vs apoptotic-like, showing mixed apoptotic features tendency to group together is evaluated)
- **Voronoi-based neighborhood graph** with a biologically motivated distance cutoff
- **Permutation testing (bootstrapping)** for statistical significance of SSi and SPI.
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
│   └── statistical_signifigance/               # permutation tests + p-values
├── src/                        # main scripts of SSI and SPI quantifications
├── data/   
│   ├──  mixed_death_annotations  # csvs of manual annotation of death times and modes
│   └── metadata.csv                # metadata of expreiments that were manually annotated in mixed_death_annotations   
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
conda conda env create -f environment.yml
conda activate CellDS

# 3) Install dependencies
pip install -r requirements.txt

# 4) Install package in editable mode
pip install -e .
```

### Configuration

Experiments are configured via YAML files in `configs/`.

* Start from `configs/default_config.yaml` to see all supported parameters
* Create your own config: `configs/<YOUR_CONFIG>.yaml` (override only what you need)
* Set the required `base_dir=<YOUR_PATH>` to point to your data location

Common paths:

* `output_dir` (default: `base_dir/anomaly_output/`) — saved representations/outputs
* `res_dir` (default: `base_dir/results/`) — evaluation results

Precedence (highest → lowest):

1. CLI args
2. Custom config YAML
3. Default config YAML

### Run

```bash
# Train anomaly detection model
python main.py --flow train --exp_name <EXP_NAME> --config configs/<CONFIG>.yaml

# Evaluate results (e.g., replication %, MoA classification, SHAP explanations)
python main.py --flow eval --exp_name <EXP_NAME>
```

---

## Citation

If you use this code in academic work, please cite the associated paper.

**Journal version (recommended):**

```bibtex
@article{shpigler2025anomaly,
  title   = {Anomaly detection for high-content image-based phenotypic cell profiling},
  author  = {Shpigler, Alon and Kolet, Naor and Golan, Shahar and Weisbart, Erin and Zaritsky, Assaf},
  journal = {Cell Systems},
  year    = {2025},
  doi     = {10.1016/j.cels.2025.101429}
}
```

**Preprint:**

```bibtex
@article{shpigler2024anomaly,
  title   = {Anomaly detection for high-content image-based phenotypic cell profiling},
  author  = {Shpigler, Alon and Kolet, Naor and Golan, Shahar and Weisbart, Erin and Zaritsky, Assaf},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.06.01.595856}
}
```

---

## License

This repository (including data, documentation, and figures where applicable) is intended for academic/research use and is released under **CC BY-NC 4.0**. See `LICENSE` for details.

---

## About

**AnomalyDetectionScreening** is a framework for hit identification in **High Throughput Screening (HTS)** using anomaly-based representations for image-based cellular profiling.

```

Sources used for accuracy (repo description/commands/structure + paper metadata): :contentReference[oaicite:0]{index=0}
::contentReference[oaicite:1]{index=1}
```
