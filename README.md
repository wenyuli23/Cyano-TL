# Cyano-TL

Code repository for paper "Multimodal AI for Photobiorefinery (MAP): Knowledge Integration, Predictive Modeling, and Techno-Economic Analysis"

## Overview

This repository contains the implementation of transfer learning methods for enhancing bioproduction predictions in cyanobacteria. The project focuses on leveraging data from source cyanobacterial species (Synechococcus sp. PCC 7942 and Synechocystis sp. PCC 6803) to improve predictions for a target species (Synechococcus elongatus UTEX 2973) in photobiological CO2 capture and conversion processes.

## Project Structure

```
├── code/                   # Source code files
│   ├── run_tl.py          # Main script for running transfer learning experiments
│   ├── rf_transfer.py     # Random Forest transfer learning implementation
│   ├── predict_tea.py     # Script for making predictions on TEA data
│   └── utils.py           # Utility functions for plotting and data processing
├── data/                  # Dataset files
│   ├── 2973.csv          # Target species data (Synechococcus elongatus UTEX 2973)
│   ├── 6803.csv          # Source species data (Synechocystis sp. PCC 6803)
│   ├── 7942.csv          # Source species data (Synechococcus sp. PCC 7942)
│   ├── impute.xlsx       # Imputation rules for missing data
│   ├── lycopene1000_nonredundant.xlsx  # Techno-Economic Analysis (TEA) data
│   └── predictions/       # Output directory for predictions
├── figures/               # Generated plots and visualizations
├── models/               # Saved trained models
└── README.md
```

## Features

The repository implements transfer learning for predicting four key bioproduction metrics:
- **Growth Rate**: Cell growth rate under different conditions
- **Optical Density (OD)**: Cell concentration measurements
- **Product Titer**: Concentration of target product produced
- **Production Rate**: Rate of product formation

## Installation

### Prerequisites
- Python 3.10
- Required packages (install via pip):

```bash
pip install pandas numpy scikit-learn matplotlib shap tqdm openpyxl joblib
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/wenyuli23/Cyano-TL.git
cd Cyano-TL
```

2. Ensure your Python environment has all required dependencies installed.

## Usage

### Running Transfer Learning Experiments

The main script `run_tl.py` provides a comprehensive interface for running transfer learning experiments:

```bash
cd code
python run_tl.py -split [percent|paper] -source [7942|6803|both] -plot [yes|no] -refit [r2|maape|mse]
```

#### Parameters:
- `split`: Data splitting method
  - `percent`: Random split (with potential information leakage)
  - `paper`: Split by paper source (no information leakage)
- `source`: Source species for transfer learning
  - `7942`: Use Synechococcus sp. PCC 7942 data
  - `6803`: Use Synechocystis sp. PCC 6803 data
  - `both`: Use combined data from both species
- `plot`: Whether to generate visualizations
  - `yes`: Generate SHAP plots and scatter plots
  - `no`: Skip plotting
- `refit`: Optimization metric for model selection
  - `r2`: R-squared score
  - `mse`: Mean Squared Error
  - `maape`: Mean Arctangent Absolute Percentage Error

#### Example:
```bash
python run_tl.py -split paper -source both -plot yes -refit r2
```

### Output Files

The experiments generate several types of output:

1. **Models**: Trained models saved as `.plk` files in `models/`
2. **Results**: CSV files with detailed results in `data/`
3. **Visualizations**: SVG plots in `figures/` including:
   - Scatter plots comparing predicted vs actual values
   - SHAP feature importance plots
4. **Predictions**: CSV files with predictions for new data in `data/predictions/`

## Citation

If you use this code in your research, please cite:

```
[Paper citation to be added upon publication]

```
