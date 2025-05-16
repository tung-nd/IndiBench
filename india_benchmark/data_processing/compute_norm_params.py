import argparse
import os
import xarray as xr
import h5py
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def process_chunk(chunk_files, variables):
    """Process a single chunk of files in parallel."""
    chunk_data = {v: [] for v in variables}
    
    for file in chunk_files:
        with h5py.File(file, "r") as f:
            for var in variables:
                chunk_data[var].append(f[var][:])
    
    chunk_data = {v: np.stack(chunk_data[v], axis=0) for v in variables}
    log_chunk_data = {v: np.log(1 + chunk_data[v] / 1e-5) for v in variables}
    diff_data = {v: np.diff(chunk_data[v], axis=0) for v in variables}
    
    chunk_stats = {
        "mean": {},
        "std": {},
        "log_mean": {},
        "log_std": {},
        "diff_mean": {},
        "diff_std": {},
    }
    
    for v in variables:
        chunk_stats["mean"][v] = chunk_data[v].mean()
        chunk_stats["std"][v] = chunk_data[v].std()
        chunk_stats["log_mean"][v] = log_chunk_data[v].mean()
        chunk_stats["log_std"][v] = log_chunk_data[v].std()
        chunk_stats["diff_mean"][v] = diff_data[v].mean()
        chunk_stats["diff_std"][v] = diff_data[v].std()
    
    return chunk_stats


def main(args):
    # Get list of h5 files
    train_root_dir = os.path.join(args.root_dir, "train")
    all_h5_files = glob(os.path.join(train_root_dir, "*.h5"))
    all_h5_files = sorted(all_h5_files)
    num_full_chunks = len(all_h5_files) // args.chunk_size
    full_chunk_files = all_h5_files[:num_full_chunks * args.chunk_size]
    
    if len(full_chunk_files) < len(all_h5_files):
        print(f"Ignoring last {len(all_h5_files) - len(full_chunk_files)} files to ensure equal chunk sizes")
    
    # load a sample file to get the variable names
    with h5py.File(all_h5_files[0], "r") as f:
        variables = [v for v in f.keys() if v != "time"]
    
    # Create chunks
    chunks = []
    for i in range(0, len(full_chunk_files), args.chunk_size):
        chunks.append(full_chunk_files[i:i+args.chunk_size])
    
    # Initialize stats dictionary
    stats_dict = {
        "mean": {v: [] for v in variables},
        "std": {v: [] for v in variables},
        "log_mean": {v: [] for v in variables},
        "log_std": {v: [] for v in variables},
        "diff_mean": {v: [] for v in variables},
        "diff_std": {v: [] for v in variables},
    }
    
    # Determine number of workers
    if args.num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_workers = args.num_workers
    
    print(f"Processing {len(chunks)} chunks using {num_workers} workers...")
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_chunk, chunk, variables): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Processing chunks"):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_stats = future.result()
                
                # Aggregate statistics
                for stat_type in stats_dict.keys():
                    for var in variables:
                        stats_dict[stat_type][var].append(chunk_stats[stat_type][var])
                
            except Exception as exc:
                print(f"Chunk {chunk_idx} generated an exception: {exc}")
    
    # summarize the statistics
    final_stats = {
        "mean": {},
        "std": {},
        "log_mean": {},
        "log_std": {},
        "diff_mean": {},
        "diff_std": {}
    }
    
    n = args.chunk_size
    for var in variables:
        # Mean calculation
        final_stats["mean"][var] = np.mean(stats_dict["mean"][var])
        final_stats["log_mean"][var] = np.mean(stats_dict["log_mean"][var])
        final_stats["diff_mean"][var] = np.mean(stats_dict["diff_mean"][var])
        
        # For standard deviation, we need to account for both within-chunk variance and between-chunk variance
        # Calculate the overall variance using the correct formula:
        # overall_var = mean_of_variances + variance_of_means
        
        # Standard data
        within_var = np.mean(np.square(stats_dict["std"][var]))  # mean of variances
        between_var = np.var(stats_dict["mean"][var])  # variance of means
        final_stats["std"][var] = np.sqrt(within_var + between_var * (n-1)/n)
        
        # Log data
        within_var_log = np.mean(np.square(stats_dict["log_std"][var]))
        between_var_log = np.var(stats_dict["log_mean"][var])
        final_stats["log_std"][var] = np.sqrt(within_var_log + between_var_log * (n-1)/n)
        
        # Diff data
        within_var_diff = np.mean(np.square(stats_dict["diff_std"][var]))
        between_var_diff = np.var(stats_dict["diff_mean"][var])
        final_stats["diff_std"][var] = np.sqrt(within_var_diff + between_var_diff * (n-1)/n)

    for stat_type, values in final_stats.items():
        final_stats[stat_type] = {var: float(val) for var, val in values.items()}
    
    # Get lat/lon information from the zarr dataset
    ds = xr.open_zarr(args.zarr_path)
    final_stats["lat"] = ds.latitude.values.tolist()
    final_stats["lon"] = ds.longitude.values.tolist()
    
    # Save to file
    output_file = os.path.join(args.root_dir, "norm_params_new.json")
    with open(output_file, "w") as f:
        json.dump(final_stats, f, indent=4)
    print(f"Normalization parameters saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalization parameters for the India dataset.")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to the zarr dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the h5 dataset.")
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size to compute mean and std over.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    main(args)