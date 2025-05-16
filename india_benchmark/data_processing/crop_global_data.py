import os
import h5py
import numpy as np
import torch
import json
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_file(file, lat_start, lat_end, lon_start, lon_end, save_dir, split):
    """Process a single H5 file by cropping it and saving to the destination"""
    try:
        data_dict = {}
        with h5py.File(file, "r") as f:
            data = f["input"]
            for var in data.keys():
                if var != 'time':
                    data_dict[var] = data[var][lat_start:lat_end, lon_start:lon_end]
                else:
                    data_dict[var] = str(data[var])
        
        output_file = os.path.join(save_dir, split, os.path.basename(file))
        with h5py.File(output_file, "w", libver="latest") as f:
            for key, array in data_dict.items():
                if key != 'time':
                    f.create_dataset(key, data=array, compression=None, dtype=np.float32)
                else:
                    f.create_dataset(key, data=array, compression=None)
        return True
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return False

def main():
    global_root_dir = '/eagle/MDClimSim/tungnd/data/wb2/0.25deg_1_step_6hr_h5df_coupling_imdaa'
    local_root_dir = '/eagle/MDClimSim/tungnd/data/imdaa/imdaa_bench_h5'
    save_dir = '/eagle/MDClimSim/tungnd/data/wb2/0.25deg_1_step_6hr_h5df_cropped_for_imdaa'
    padding_factor = 0.0

    # Load coordinates
    lat = np.load(os.path.join(global_root_dir, "lat.npy"))
    global_lat = torch.tensor(lat)
    lon = np.load(os.path.join(global_root_dir, "lon.npy"))
    global_lon = torch.tensor(lon)
    global_lat_descending = (global_lat[0] > global_lat[-1])

    with open(os.path.join(local_root_dir, "norm_params.json"), "r") as f:
        norm_params = json.load(f)
        local_lat = torch.tensor(norm_params["lat"])
        local_lon = torch.tensor(norm_params["lon"])

    # Get local region boundaries
    local_lat_min, local_lat_max = local_lat.min(), local_lat.max()
    local_lon_min, local_lon_max = local_lon.min(), local_lon.max()

    # Add padding
    lat_range = local_lat_max - local_lat_min
    lon_range = local_lon_max - local_lon_min

    padded_lat_min = local_lat_min - padding_factor * lat_range
    padded_lat_max = local_lat_max + padding_factor * lat_range
    padded_lon_min = local_lon_min - padding_factor * lon_range
    padded_lon_max = local_lon_max + padding_factor * lon_range

    # Find closest indices in global grid based on latitude order
    if global_lat_descending:
        # Global latitudes are in descending order (e.g., 90 to -90)
        lat_start = torch.abs(global_lat - padded_lat_max).argmin()  # Note: max becomes start in descending order
        lat_end = torch.abs(global_lat - padded_lat_min).argmin() + 1  # min becomes end in descending order
        
        # Ensure correct order (start should be less than end for slicing)
        if lat_start > lat_end:
            lat_start, lat_end = lat_end - 1, lat_start + 1
    else:
        # Global latitudes are in ascending order (same as local)
        lat_start = torch.abs(global_lat - padded_lat_min).argmin()
        lat_end = torch.abs(global_lat - padded_lat_max).argmin() + 1

    # Longitudes are always in ascending order
    lon_start = torch.abs(global_lon - padded_lon_min).argmin()
    lon_end = torch.abs(global_lon - padded_lon_max).argmin() + 1

    lat_start = lat_start.item()
    lat_end = lat_end.item()
    lon_start = lon_start.item()
    lon_end = lon_end.item()

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(save_dir, split), exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")
        global_split_files = sorted(glob(os.path.join(global_root_dir, split, "*.h5")))
        
        # Create a partial function with fixed parameters
        process_func = partial(
            process_file, 
            lat_start=lat_start, 
            lat_end=lat_end, 
            lon_start=lon_start, 
            lon_end=lon_end,
            save_dir=save_dir,
            split=split
        )
        
        # Determine optimal number of processes (use 75% of available cores)
        num_cpus = max(1, int(mp.cpu_count() * 0.55))
        print(f"Using {num_cpus} CPU cores")
        
        # Process files in parallel with a progress bar
        with mp.Pool(processes=num_cpus) as pool:
            results = list(tqdm(
                pool.imap(process_func, global_split_files),
                total=len(global_split_files),
                desc=f"Processing {split}"
            ))
        
        # Report results
        success_count = results.count(True)
        print(f"Completed {split}: {success_count}/{len(global_split_files)} files processed successfully")

if __name__ == "__main__":
    # This is important for multiprocessing to work correctly
    main()