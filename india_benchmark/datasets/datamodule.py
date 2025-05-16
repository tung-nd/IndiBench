from typing import Optional
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

# Local application
from india_benchmark.datasets.india_dataset import IndiaDataset

def collate_fn_with_filenames(batch):
    X = torch.stack([b[0] for b in batch])
    Y = torch.stack([b[1] for b in batch])
    filenames = [b[2] for b in batch]
    return X, Y, filenames

class IndiaDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        variables,
        lead_time=6,
        data_freq=6,
        n_input_steps=1,
        n_output_steps=1,
        n_test_outoput_steps=1,
        predict_diff=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        return_filename=False,
    ):
        super().__init__()

        self.save_hyperparameters()
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset Path: {root_dir} does not exist.")

        self.load_norm_params(os.path.join(root_dir, "norm_params.json"))

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def load_norm_params(self, norm_params_path):
        with open(norm_params_path, "r") as f:
            norm_params = json.load(f)
            mean_dict = norm_params["mean"]
            std_dict = norm_params["std"]
            diff_mean_dict = norm_params["diff_mean"]
            diff_std_dict = norm_params["diff_std"]
            self.lat = torch.tensor(norm_params["lat"])
            self.lon = torch.tensor(norm_params["lon"])
        
        means, stds = [], []
        diff_means, diff_stds = [], []
        for var in self.hparams.variables:
            means.append(mean_dict[var])
            stds.append(std_dict[var])
            diff_means.append(diff_mean_dict[var])
            diff_stds.append(diff_std_dict[var])
        self.transform_mean = torch.tensor(means)
        self.transform_std = torch.tensor(stds)
        self.transform_diff_mean = torch.tensor(diff_means)
        self.transform_diff_std = torch.tensor(diff_stds)
    
    def reshape_norm_params(self, mean, std, shape_dim):
        if shape_dim == 5:  # (B, T, C, H, W)
            mean, std = mean[None, None, :, None, None], std[None, None, :, None, None]
        elif shape_dim == 4:  # (B, C, H, W)
            mean, std = mean[None, :, None, None], std[None, :, None, None]
        else:  # (B, N, C)
            mean, std = mean[None, None, :], std[None, None, :]
        return mean, std
    
    def normalize(self, sample):
        mean, std = self.transform_mean.to(dtype=sample.dtype, device=sample.device), self.transform_std.to(dtype=sample.dtype, device=sample.device)  # (C,)
        mean, std = self.reshape_norm_params(mean, std, sample.dim())
        return (sample - mean) / std
    
    def denormalize(self, sample):
        mean, std = self.transform_mean.to(dtype=sample.dtype, device=sample.device), self.transform_std.to(dtype=sample.dtype, device=sample.device)
        mean, std = self.reshape_norm_params(mean, std, sample.dim())
        return sample * std + mean

    def normalize_diff(self, diff):
        diff_mean, diff_std = self.transform_diff_mean.to(dtype=diff.dtype, device=diff.device), self.transform_diff_std.to(dtype=diff.dtype, device=diff.device)
        diff_mean, diff_std = self.reshape_norm_params(diff_mean, diff_std, diff.dim())
        return (diff - diff_mean) / diff_std

    def denormalize_diff(self, diff):
        diff_mean, diff_std = self.transform_diff_mean.to(dtype=diff.dtype, device=diff.device), self.transform_diff_std.to(dtype=diff.dtype, device=diff.device)
        diff_mean, diff_std = self.reshape_norm_params(diff_mean, diff_std, diff.dim())
        return diff * diff_std + diff_mean

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            norm_params = {
                "mean": self.transform_mean,
                "std": self.transform_std,
                "diff_mean": self.transform_diff_mean,
                "diff_std": self.transform_diff_std,
            }
            self.data_train = IndiaDataset(
                dataset_path=os.path.join(self.hparams.root_dir, "train"),
                variables=self.hparams.variables,
                norm_params=norm_params,
                lead_time=self.hparams.lead_time,
                data_freq=self.hparams.data_freq,
                n_input_steps=self.hparams.n_input_steps,
                n_output_steps=self.hparams.n_output_steps,
                predict_diff=self.hparams.predict_diff,
                return_filename=self.hparams.return_filename,
            )
            
            self.data_val = IndiaDataset(
                dataset_path=os.path.join(self.hparams.root_dir, "val"),
                variables=self.hparams.variables,
                norm_params=norm_params,
                lead_time=self.hparams.lead_time,
                data_freq=self.hparams.data_freq,
                n_input_steps=self.hparams.n_input_steps,
                n_output_steps=self.hparams.n_test_outoput_steps,
                predict_diff=False,
                return_filename=self.hparams.return_filename,
            )

            self.data_test = IndiaDataset(
                dataset_path=os.path.join(self.hparams.root_dir, "test"),
                variables=self.hparams.variables,
                norm_params=norm_params,
                lead_time=self.hparams.lead_time,
                data_freq=self.hparams.data_freq,
                n_input_steps=self.hparams.n_input_steps,
                n_output_steps=self.hparams.n_test_outoput_steps,
                predict_diff=False,
                return_filename=self.hparams.return_filename,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_with_filenames if self.hparams.return_filename else None,
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
                collate_fn=collate_fn_with_filenames if self.hparams.return_filename else None,
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
                collate_fn=collate_fn_with_filenames if self.hparams.return_filename else None,
            )

# datamodule = IndiaDataModule(
#     root_dir='/eagle/MDClimSim/tungnd/data/imdaa/imdaa_bench_h5',
#     variables=['TMP', 'APCP', 'TMP_prl925', 'TMP_prl850', 'UGRD_prl925', 'UGRD_prl850', 'VGRD_prl925', 'VGRD_prl850'],
#     lead_time=6,
#     data_freq=6,
#     n_input_steps=2,
#     n_output_steps=3,
#     batch_size=2,
#     num_workers=1,
#     pin_memory=False,
# )
# datamodule.setup()
# train_loader = datamodule.train_dataloader()
# x, y, _ = next(iter(train_loader))
# print(x.shape, y.shape)
# print(x)