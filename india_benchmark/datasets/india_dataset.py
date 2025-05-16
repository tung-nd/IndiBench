import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob


class IndiaDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        variables,
        norm_params,
        lead_time=6,
        data_freq=6,
        n_input_steps=1,
        n_output_steps=1,
        predict_diff=False,
        return_filename=False,
    ):
        """
        Args:
            dataset_path (str): Path to the Zarr dataset.
            variables (list): List of variables to extract (e.g., ["temp", "precip"]).
            start_date (str): Start date for the dataset (e.g., "2000-01-01").
            end_date (str): End date for the dataset (e.g., "2010-12-31").
            lat_range (slice): Latitude range to crop. Default is slice(6, 36.72).
            lon_range (slice): Longitude range to crop. Default is slice(66.5, 97.25).
            spatial_resolution (float): Spatial resolution for grid. Default is 0.12.
            lead_time (int): Lead time (in hours) to define the target sample.
            data_transform (callable): Transformation function applied to the data.
        """
        self.variables = variables
        self.transform_mean = norm_params['mean']
        self.transform_std = norm_params['std']
        self.transform_diff_mean = norm_params['diff_mean']
        self.transform_diff_std = norm_params['diff_std']
        self.lead_time = lead_time
        self.data_freq = data_freq
        self.lead_time_steps = lead_time // data_freq
        self.n_input_steps = n_input_steps
        self.n_output_steps = n_output_steps
        self.predict_diff = predict_diff
        self.return_filename = return_filename
        h5_files = glob(os.path.join(dataset_path, "*.h5"))
        self.h5_files = sorted(h5_files)
    
    def normalize(self, sample):
        # sample: (n, channel, latitude, longitude)
        return (sample - self.transform_mean[None, :, None, None]) / self.transform_std[None, :, None, None]
    
    def normalize_diff(self, sample):
        # sample: (n, channel, latitude, longitude)
        return (sample - self.transform_diff_mean[None, :, None, None]) / self.transform_diff_std[None, :, None, None]
   
    def get_ids(self, ids, transform=True):
        all_frames = []
        for i in ids:
            frame_data = []
            with h5py.File(self.h5_files[i], "r") as f:
                for var in self.variables:
                    frame_data.append(f[var][()])
            all_frames.append(np.stack(frame_data, axis=0))
        all_frames = torch.from_numpy(np.stack(all_frames, axis=0))
        return self.normalize(all_frames) if transform else all_frames

    def __len__(self):
        # Ensure we have a valid pair (current, future) for each sample.
        return len(self.h5_files) - self.lead_time_steps * (self.n_input_steps + self.n_output_steps - 1)

    def __getitem__(self, idx):
        """
        Allows random access by returning a pair (X, Y), where X is the data at the current time
        and Y is the data at the time offset by the lead time.
        """
        ids = range(idx, idx + self.n_input_steps + self.n_output_steps)
        sample_seq = self.get_ids(ids, transform=False)
        filenames = [self.h5_files[i] for i in ids]
        X = torch.as_tensor(sample_seq[:self.n_input_steps]).float()
        X = self.normalize(X)
        
        if not self.predict_diff:
            Y = torch.as_tensor(sample_seq[self.n_input_steps::]).float()
            Y = self.normalize(Y)
        else:
            Y = torch.as_tensor(sample_seq[self.n_input_steps:] - sample_seq[self.n_input_steps - 1:-1]).float()
            Y = self.normalize_diff(Y)

        if self.return_filename:
            return X, Y, filenames
        else:
            return X, Y


# dataset = IndiaDataset(
#     dataset_path="/eagle/MDClimSim/tungnd/data/imdaa/imdaa_bench_incremental.zarr",
#     variables=['APCP', 'TMP', 'TMP_prl','UGRD', 'UGRD_prl','VGRD','PRMSL'],
#     start_date="2000-01-01",
#     end_date="2017-12-31",
#     lead_time=6,
# )
# with np.load('/eagle/MDClimSim/tungnd/data/imdaa/normalization_parameters.npz', allow_pickle=True) as data:
#     mean_dict = data["mean"].item()
#     std_dict = data["std"].item()
#     log_mean_dict = data["log_mean"].item()
#     log_std_dict = data["log_std"].item()
# dataset.set_norm_params(mean_dict, std_dict, log_mean_dict, log_std_dict)
# print(dataset.channel_mapping)
# print(dataset.transform_channel_mapping)
# print('norm params: ', dataset.transform_mean, dataset.transform_std)
# print('log norm params: ', dataset.transform_log_mean, dataset.transform_log_std)
# x, y = dataset[0]
# print(x.shape, y.shape)
# print('normalized x: ', x[:, 0, :])
# print('denormalized x: ', dataset.denormalize(x)[:, 0, :])
# for c in dataset.channel_mapping:
#     print(c)
#     print(x[dataset.channel_mapping[c]])
#     print('='*50)