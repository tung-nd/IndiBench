# IndiaWeatherBench: A Benchmark for Regional Weather Forecasting over the Indian Subcontinent

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets/tungnd/IndiaWeatherBench)

## 🌏 Overview

**IndiaWeatherBench** is a comprehensive benchmark for data-driven regional weather forecasting focused on the Indian subcontinent. While machine learning has shown impressive progress in global weather forecasting, regional forecasting remains comparatively underexplored. This benchmark provides a curated dataset built from high-resolution regional reanalysis products (specifically IMDAA), along with a suite of strong baselines and standard metrics to facilitate consistent training and evaluation of ML models for regional weather prediction.

## 📦 Dataset

IndiaWeatherBench provides a curated benchmark dataset built from the Indian Monsoon Data Assimilation and Analysis (IMDAA) reanalysis dataset, which was produced under the National Monsoon Mission by NCMRWF, UK Met Office, and IMD.

- **Time range**: 2000–2019 (20 years)
- **Interval**: 6-hourly (00, 06, 12, 18 UTC)
- **Region**: 6°N–36.72°N, 66.6°E–97.25°E (~256×256 grid)
- **Train/Val/Test splits**:
  - Train: 2000–2017 (~26,500 samples)
  - Val: 2018 (~1,500 samples)
  - Test: 2019 (~1,500 samples)
- **Variables**: 39 channels across the following categories:
  - Single-level: TMP (2m temp), UGRD/VGRD (10m wind), APCP (precip), PRMSL (MSLP), TCDCRO (cloud cover)
  - Pressure-level: TMP_prl, HGT, UGRD_prl, VGRD_prl, RH — at 50, 250, 500, 600, 700, 850, 925 hPa
  - Static fields: MTERH (terrain height), LAND (land cover)

### Data Formats

IndiaWeatherBench is released in two formats:

#### Zarr Format
- Chunked, cloud-native array storage
- Compatible with xarray and dask
- Suitable for scientific analysis and fast slicing

```python
import xarray as xr
ds = xr.open_zarr("imdaa_bench_incremental.zarr", consolidated=True)
```

#### HDF5 Format
- Optimized for ML training
- Each .h5 file = one time step with all variables
- Pre-split into train/, val/, and test/ directories

```python
import h5py
f = h5py.File("imdaa_bench_h5/train/20010101_00.h5", "r")
print(list(f.keys()))
```

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/tung-nd/IndiaWeatherBench.git
cd IndiaWeatherBench

# Create conda environment
conda env create -f environment.yml
conda activate IndiaWeatherBench

# Install package in development mode
pip install -e .
```

## 🚀 Usage

### Training Models

IndiaWeatherBench includes implementations of four architectures - UNet, Stormer, Graphcast, and Hierarchical Graphcast, with two boundary conditioning strategies: boundary forcing and coarse-resolution conditioning. For example, to train Graphcast with boundary forcing, run:

```bash
python train_boundary_forcing.py --config configs/boundary_forcing_gc.yaml
```

Please look at the config file to understand the configurable parameters. Users can override the default parameters via the CLI. For example, if you want to change the number of input steps from 2 to 1, you can specify `--data.n_input_steps=1`. 

### Evaluation

After training, you can evaluate a saved checkpoint on the test data. For example, to test Graphcast with boundary forcing, run:

```bash
# Evaluate a trained model
python test_boundary_forcing.py --config configs/boundary_forcing_gc.yaml --ckpt_path [MODEL_CHECKPOINT]
```

## 📜 License and Terms of Use

The raw `.zarr` dataset and its derived `h5df` version are released under the Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

- ✅ Free for non-commercial, educational, and research use
- ❌ For commercial use, contact: director@ncmrwf.gov.in
- 📧 Send a copy of any publication using this dataset to the same address

## 📬 Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the authors.