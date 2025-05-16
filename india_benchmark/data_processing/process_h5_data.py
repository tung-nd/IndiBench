import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from india_benchmark.utils.data_utils import (
    SURFACE_VARIABLES,
    PRESSURE_VARIABLES,
    CONSTANTS
)


def process_single_date(
    dataset_path,
    date_str,
    variables,
    pressure_levels,
    save_dir,
    split,
    spatial_resolution=0.12,
):
    """Process all time steps for a single date in sequential order."""
    save_dir_split = os.path.join(save_dir, split)
    os.makedirs(save_dir_split, exist_ok=True)
    
    list_constant_vars = [v for v in variables if v in CONSTANTS]
    list_single_vars = [v for v in variables if v in SURFACE_VARIABLES and v not in CONSTANTS]
    list_pressure_vars = [v for v in variables if v in PRESSURE_VARIABLES]
    
    # Open dataset and select only the specific date
    ds = xr.open_zarr(dataset_path)
    date_start = f"{date_str}T00:00:00"
    date_end = f"{date_str}T23:59:59"
    ds = ds.sel(time=slice(date_start, date_end))
    
    # Apply spatial resolution adjustment if needed
    if spatial_resolution != 0.12:
        num_grid_cells = int(256 / (spatial_resolution // 0.12))
        lat_values = np.linspace(ds.latitude.values[0], ds.latitude.values[-1], num_grid_cells)
        lon_values = np.linspace(ds.longitude.values[0], ds.longitude.values[-1], num_grid_cells)
        ds = ds.sel(
            latitude=xr.DataArray(lat_values, dims="latitude"),
            longitude=xr.DataArray(lon_values, dims="longitude"),
            method="nearest"
        )
    
    idx_in_date = 0
    for idx in range(len(ds.time)):
        ds_idx = ds.isel(time=idx)
        time_stamp = pd.to_datetime(ds_idx.time.values)
        date = time_stamp.date()
        
        data_dict = {
            'time': str(time_stamp)
        }
        for var in (list_single_vars + list_constant_vars):
            data_dict[var] = ds_idx[var].values
        for var in list_pressure_vars:
            for level in pressure_levels:
                data_dict[f'{var}{int(level)}'] = ds_idx[var].sel(isobaricInhPa=level).values
        
        # save to h5 file
        h5_path = os.path.join(save_dir_split, f'{str(date)}_{idx_in_date:02}.h5')
        with h5py.File(h5_path, 'w', libver='latest') as f:
            for key, array in data_dict.items():
                if key != 'time':
                    f.create_dataset(key, data=array, compression=None, dtype=np.float32)
                else:
                    f.create_dataset(key, data=array, compression=None)
        
        idx_in_date += 1
    
    return f"Processed {len(ds.time)} timestamps for date {date_str}"


def create_parallel_dataset(
    dataset_path,
    variables,
    pressure_levels,
    start_date,
    end_date,
    save_dir,
    split,
    spatial_resolution=0.12,
    max_workers=None,
):
    """Process all dates in the range in parallel."""
    # Generate list of dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strings = [date.strftime('%Y-%m-%d') for date in date_range]
    
    # Process dates in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(
                process_single_date,
                dataset_path,
                date_str,
                variables,
                pressure_levels,
                save_dir,
                split,
                spatial_resolution
            ): date_str for date_str in date_strings
        }
        
        # Show progress as tasks complete
        for future in tqdm(as_completed(future_to_date), total=len(date_strings), desc="Processing dates"):
            date_str = future_to_date[future]
            try:
                _ = future.result()
            except Exception as exc:
                print(f"{date_str} generated an exception: {exc}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Zarr dataset.")
    parser.add_argument("--variables", type=str, nargs="+", required=True, help="List of variables to extract.")
    parser.add_argument("--pressure_levels", type=int, nargs="+", required=True, help="List of pressure levels.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for the dataset.")
    parser.add_argument("--end_date", type=str, required=True, help="End date for the dataset.")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset (train, val, test).")
    parser.add_argument("--spatial_resolution", type=float, default=0.12, help="Spatial resolution for grid.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    create_parallel_dataset(
        dataset_path=args.dataset_path,
        variables=args.variables,
        pressure_levels=args.pressure_levels,
        start_date=args.start_date,
        end_date=args.end_date,
        save_dir=args.save_dir,
        split=args.split,
        spatial_resolution=args.spatial_resolution,
        max_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()