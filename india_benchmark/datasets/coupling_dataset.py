import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob


class CouplingDataset(Dataset):
    def __init__(
        self,
        global_dataset_path,
        global_variables,
        global_norm_params,
        global_lat,
        global_lon,
        local_dataset_path,
        local_variables,
        local_norm_params,
        local_lat,
        local_lon,
        years,
        lead_time=6,
        data_freq=6,
        n_input_steps=1,
        n_output_steps=1,
        padding_factor=0.0,  # Add padding around local region (as a fraction of local region size)
        ignore_last_local_files=1, # due to current era5 missing last time step of each year
    ):
        self.global_variables = global_variables
        self.global_transform_mean = global_norm_params['mean']
        self.global_transform_std = global_norm_params['std']
        self.local_variables = local_variables
        self.local_transform_mean = local_norm_params['mean']
        self.local_transform_std = local_norm_params['std']
        self.lead_time = lead_time
        self.data_freq = data_freq
        self.lead_time_steps = lead_time // data_freq
        self.n_input_steps = n_input_steps
        self.n_output_steps = n_output_steps
        self.padding_factor = padding_factor
        
        self.global_lat = global_lat
        self.global_lon = global_lon
        self.local_lat = local_lat
        self.local_lon = local_lon
        self.global_lat_descending = (self.global_lat[0] > self.global_lat[-1])
        # self.crop_indices = self._compute_crop_indices()
        
        years = sorted(years)
        
        global_h5_files = glob(os.path.join(global_dataset_path, "*.h5"))
        global_h5_files = sorted(global_h5_files)
        global_h5_files = [f for f in global_h5_files if any(f"{year}" in f for year in years)]
        self.global_h5_files = sorted(global_h5_files)
        
        local_h5_files = []
        for year in years:
            year_files = glob(os.path.join(local_dataset_path, f"*{year}*.h5"))
            year_files = sorted(year_files)
            if ignore_last_local_files > 0:
                year_files = year_files[:-ignore_last_local_files]
            local_h5_files.extend(year_files)
        self.local_h5_files = sorted(local_h5_files)
        
        assert len(self.global_h5_files) == len(self.local_h5_files), \
            f"Number of global files ({len(self.global_h5_files)}) does not match number of local files ({len(self.local_h5_files)})."
    
    def normalize_global(self, sample):
        # sample: (n, channel, latitude, longitude)
        return (sample - self.global_transform_mean[None, :, None, None]) / self.global_transform_std[None, :, None, None]
    
    def normalize_local(self, sample):
        # sample: (n, channel, latitude, longitude)
        return (sample - self.local_transform_mean[None, :, None, None]) / self.local_transform_std[None, :, None, None]
   
    def get_local(self, ids, transform=True):
        all_frames = []
        for i in ids:
            frame_data = []
            with h5py.File(self.local_h5_files[i], "r") as f:
                for var in self.local_variables:
                    frame_data.append(f[var][()])
            all_frames.append(np.stack(frame_data, axis=0))
        all_frames = torch.from_numpy(np.stack(all_frames, axis=0))
        return self.normalize_local(all_frames) if transform else all_frames
    
    def get_global(self, ids, transform=True):
        all_frames = []
        for i in ids:
            frame_data = []
            with h5py.File(self.global_h5_files[i], "r") as f:
                for var in self.global_variables:
                    # Crop the global data to the local region
                    data = f[var][()]
                    frame_data.append(data)
            frame_data = np.stack(frame_data, axis=0)
            if self.global_lat_descending:
                frame_data = np.flip(frame_data, axis=1)
            all_frames.append(frame_data)
        all_frames = torch.from_numpy(np.stack(all_frames, axis=0))
        return self.normalize_global(all_frames) if transform else all_frames

    def __len__(self):
        # Ensure we have a valid pair (current, future) for each sample.
        return len(self.local_h5_files) - self.lead_time_steps * (self.n_input_steps + self.n_output_steps - 1)

    def __getitem__(self, idx):
        """
        Allows random access by returning a pair (X, Y), where X is the data at the current time
        and Y is the data at the time offset by the lead time.
        """
        ids = range(idx, idx + self.n_input_steps + self.n_output_steps)
        
        local_sample = torch.as_tensor(self.get_local(ids)).float()
        local_input = local_sample[:self.n_input_steps]
        local_output = local_sample[self.n_input_steps:]
        
        # get cropped global data
        global_sample = torch.as_tensor(self.get_global(ids)).float()
        
        # interpolate cropped global data to local grid resolution
        global_sample = torch.nn.functional.interpolate(
            global_sample,
            size=(local_sample.shape[-2], local_sample.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        global_input = global_sample[:self.n_input_steps]
        global_output = global_sample[self.n_input_steps:]

        return global_input, local_input, global_output, local_output
