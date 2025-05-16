# India Forecast Benchmark

In this repository, we present code for the use of our Indian weather benchmark dataset, along with implementations of simple baselines.

## Dataset location

On Mint3: /data0/imdaa/imdaa_bench_incremental.zarr

On Mint1: /data1/imdaa/data/imdaa_bench_incremental.zarr

## Repository Map

- QuickStart.ipynb: sample code to use code repository to build forecast, evaluate, visualize, and more.
- CS269_AI_for_Climate_Final_Report.pdf: PDF of final project report
- models : PyTorch implementations of a few naive baselines (Linear, Persistence, UNet)
    - In addition, climatology.ipynb with which we calculated climatology
- utils : a few utils used to evaluate and visualize forecasts
- datasets:
    - weatherdataset.py: implementation of WeatherDataset class (inherits from torch.utils.data.IterableDataset), used for integration with pytorch data loader & model training framework
    - custom_transforms.py: data transformations for WeatherDataset