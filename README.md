# CellDeathSpreading

A **PyTorch** framework for **anomaly detection in cellular image-based profiling** to support hit identification in high-throughput screening (HTS). The core idea is to learn the distribution of “normal” (control) wells via **self-supervised reconstruction**, then use reconstruction-driven signals as **interpretable anomaly-based representations** for downstream analysis (e.g., reproducibility, MoA classification, and feature-level explanations).

### Key Features

- Self-supervised learning from control data (no anomaly labels required)
- Captures complex morphological dependencies in high-content profiles
- Interpretable anomaly signals (feature-level insight, not just scores)
- Fast training and inference (HTS-friendly workflows)
- PyTorch-based, configurable experiments via YAML

![Overview figure](figures/<ADD_FIGURE_NAME>.png)

---

## Downloading data

The project uses **augmented per-well aggregated Cell Painting** datasets hosted on the **Cell Painting Gallery (CPG)** (AWS Open Data).

Example (dataset used in the paper):

```bash
aws s3 cp --no-sign-request \
  s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/ \
  <YOUR_PATH> --recursive
````

> You’ll need the **AWS CLI** installed and available on your PATH.

---

## Setup Instructions

### Quick Start

```bash
# 1) Clone and enter repo
git clone https://github.com/zaritskylab/AnomalyDetectionScreening.git
cd AnomalyDetectionScreening

# 2) Create and activate environment
conda create -n pytorch_anomaly python=3.10.9
conda activate pytorch_anomaly

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

## Repository Structure

```text
AnomalyDetectionScreening/
├── README.md
├── main.py
├── requirements.txt
├── setup.py
├── LICENSE
├── configs/                 # YAML experiment configs
├── notebooks/               # analysis & interpretation notebooks
│   ├── analyze_moa_res.ipynb
│   └── interpret_feature_dists.ipynb
├── figures/                 # figures used in docs/paper
├── sbatch/                  # HPC job scripts (if applicable)
└── src/                     # main package
    ├── __init__.py
    ├── data/                # data utilities / loaders
    ├── model/               # model components
    ├── eval/                # evaluation pipeline
    ├── utils/               # shared helpers
    └── ProfilingAnomalyDetector.py
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
