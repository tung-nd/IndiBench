from typing import Optional
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

# Local application
from india_benchmark.datasets.coupling_dataset import CouplingDataset

class CouplingDataModule(LightningDataModule):
    def __init__(
        self,
        global_root_dir,
        global_variables,
        local_root_dir,
        local_variables,
        train_year_start,
        train_year_end,
        val_year_start,
        val_year_end,
        test_year_start,
        test_year_end,
        padding_factor=0.0,
        ignore_last_local_files=1,
        lead_time=6,
        data_freq=6,
        n_input_steps=1,
        n_output_steps=1,
        n_test_outoput_steps=1,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.load_global_norm_params()
        self.load_local_norm_params()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def load_local_norm_params(self):
        with open(os.path.join(self.hparams.local_root_dir, "norm_params.json"), "r") as f:
            norm_params = json.load(f)
            mean_dict = norm_params["mean"]
            std_dict = norm_params["std"]
            diff_mean_dict = norm_params["diff_mean"]
            diff_std_dict = norm_params["diff_std"]
            self.local_lat = torch.tensor(norm_params["lat"])
            self.local_lon = torch.tensor(norm_params["lon"])
        
        means, stds = [], []
        diff_means, diff_stds = [], []
        for var in self.hparams.local_variables:
            means.append(mean_dict[var])
            stds.append(std_dict[var])
            diff_means.append(diff_mean_dict[var])
            diff_stds.append(diff_std_dict[var])
        self.local_transform_mean = torch.tensor(means)
        self.local_transform_std = torch.tensor(stds)
        self.local_transform_diff_mean = torch.tensor(diff_means)
        self.local_transform_diff_std = torch.tensor(diff_stds)
    
    def load_global_norm_params(self):
        mean = dict(np.load(os.path.join(self.hparams.global_root_dir, "normalize_mean.npz")))
        mean = np.concatenate([mean[v] for v in self.hparams.global_variables], axis=0)
        self.global_transform_mean = torch.tensor(mean)
        std = dict(np.load(os.path.join(self.hparams.global_root_dir, "normalize_std.npz")))
        std = np.concatenate([std[v] for v in self.hparams.global_variables], axis=0)
        self.global_transform_std = torch.tensor(std)
        
        # NOTE: fix hardcoded values
        diff_mean = dict(np.load(os.path.join(self.hparams.global_root_dir, "normalize_diff_mean_6.npz")))
        diff_mean = np.concatenate([diff_mean[v] for v in self.hparams.global_variables], axis=0)
        self.global_transform_diff_mean = torch.tensor(diff_mean)
        diff_std = dict(np.load(os.path.join(self.hparams.global_root_dir, "normalize_diff_std_6.npz")))
        diff_std = np.concatenate([diff_std[v] for v in self.hparams.global_variables], axis=0)
        self.global_transform_diff_std = torch.tensor(diff_std)
        
        lat = np.load(os.path.join(self.hparams.global_root_dir, "lat.npy"))
        self.global_lat = torch.tensor(lat)
        lon = np.load(os.path.join(self.hparams.global_root_dir, "lon.npy"))
        self.global_lon = torch.tensor(lon)
    
    def reshape_norm_params(self, mean, std, shape_dim):
        if shape_dim == 5:  # (B, T, C, H, W)
            mean, std = mean[None, None, :, None, None], std[None, None, :, None, None]
        elif shape_dim == 4:  # (B, C, H, W)
            mean, std = mean[None, :, None, None], std[None, :, None, None]
        else:  # (B, N, C)
            mean, std = mean[None, None, :], std[None, None, :]
        return mean, std
    
    def get_mean_std(self, type, scope):
        if type ==  "state":
            if scope == "local":
                mean, std = self.local_transform_mean, self.local_transform_std
            else:
                mean, std = self.global_transform_mean, self.global_transform_std
        elif type == "diff":
            if scope == "local":
                mean, std = self.local_transform_diff_mean, self.local_transform_diff_std
            else:
                mean, std = self.global_transform_diff_mean, self.global_transform_diff_std
        return mean, std
    
    def normalize(self, x, type, scope):
        mean, std = self.get_mean_std(type, scope)
        mean, std = mean.to(dtype=x.dtype, device=x.device), std.to(dtype=x.dtype, device=x.device)  # (C,)
        mean, std = self.reshape_norm_params(mean, std, x.dim())
        return (x - mean) / std
    
    def denormalize(self, x, type, scope):
        mean, std = self.get_mean_std(type, scope)
        mean, std = mean.to(dtype=x.dtype, device=x.device), std.to(dtype=x.dtype, device=x.device)
        mean, std = self.reshape_norm_params(mean, std, x.dim())
        return x * std + mean

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            local_norm_params = {
                "mean": self.local_transform_mean,
                "std": self.local_transform_std,
                "diff_mean": self.local_transform_diff_mean,
                "diff_std": self.local_transform_diff_std,
            }
            global_norm_params = {
                "mean": self.global_transform_mean,
                "std": self.global_transform_std,
                "diff_mean": self.global_transform_diff_mean,
                "diff_std": self.global_transform_diff_std,
            }
            self.data_train = CouplingDataset(
                global_dataset_path=os.path.join(self.hparams.global_root_dir, "train"),
                global_variables=self.hparams.global_variables,
                global_norm_params=global_norm_params,
                global_lat=self.global_lat,
                global_lon=self.global_lon,
                local_dataset_path=os.path.join(self.hparams.local_root_dir, "train"),
                local_variables=self.hparams.local_variables,
                local_norm_params=local_norm_params,
                local_lat=self.local_lat,
                local_lon=self.local_lon,
                years=list(range(self.hparams.train_year_start, self.hparams.train_year_end + 1)),
                lead_time=self.hparams.lead_time,
                data_freq=self.hparams.data_freq,
                n_input_steps=self.hparams.n_input_steps,
                n_output_steps=self.hparams.n_output_steps,
                padding_factor=self.hparams.padding_factor,
                ignore_last_local_files=self.hparams.ignore_last_local_files,
            )
            
            self.data_val = CouplingDataset(
                global_dataset_path=os.path.join(self.hparams.global_root_dir, "val"),
                global_variables=self.hparams.global_variables,
                global_norm_params=global_norm_params,
                global_lat=self.global_lat,
                global_lon=self.global_lon,
                local_dataset_path=os.path.join(self.hparams.local_root_dir, "val"),
                local_variables=self.hparams.local_variables,
                local_norm_params=local_norm_params,
                local_lat=self.local_lat,
                local_lon=self.local_lon,
                years=list(range(self.hparams.val_year_start, self.hparams.val_year_end + 1)),
                lead_time=self.hparams.lead_time,
                data_freq=self.hparams.data_freq,
                n_input_steps=self.hparams.n_input_steps,
                n_output_steps=self.hparams.n_test_outoput_steps,
                padding_factor=self.hparams.padding_factor,
                ignore_last_local_files=self.hparams.ignore_last_local_files,
            )

            self.data_test = CouplingDataset(
                global_dataset_path=os.path.join(self.hparams.global_root_dir, "test"),
                global_variables=self.hparams.global_variables,
                global_norm_params=global_norm_params,
                global_lat=self.global_lat,
                global_lon=self.global_lon,
                local_dataset_path=os.path.join(self.hparams.local_root_dir, "test"),
                local_variables=self.hparams.local_variables,
                local_norm_params=local_norm_params,
                local_lat=self.local_lat,
                local_lon=self.local_lon,
                years=list(range(self.hparams.test_year_start, self.hparams.test_year_end + 1)),
                lead_time=self.hparams.lead_time,
                data_freq=self.hparams.data_freq,
                n_input_steps=self.hparams.n_input_steps,
                n_output_steps=self.hparams.n_test_outoput_steps,
                padding_factor=self.hparams.padding_factor,
                ignore_last_local_files=self.hparams.ignore_last_local_files,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )


# datamodule = CouplingDataModule(
#     global_root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_1_step_6hr_h5df_coupling_imdaa',
#     global_variables=["2m_temperature", "geopotential_500", "temperature_850"],
#     local_root_dir='/eagle/MDClimSim/tungnd/data/imdaa/imdaa_bench_h5',
#     local_variables=["TMP", "UGRD", "VGRD", "HGT500", "TMP_prl850"],
#     train_years=[2015, 2016, 2017],
#     val_years=[2018],
#     test_years=[2019],
#     ignore_last_local_files=1,
#     lead_time=6,
#     data_freq=6,
#     n_input_steps=2,
#     n_output_steps=1,
#     n_test_outoput_steps=4,
#     batch_size=2,
#     num_workers=1,
#     pin_memory=False,
# )
# datamodule.setup()
# train_loader = datamodule.train_dataloader()
# global_inp, local_inp, global_out, local_out = next(iter(train_loader))
# print(global_inp.shape, local_inp.shape, global_out.shape, local_out.shape)