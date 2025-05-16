import os
import xarray as xr
import numpy as np
import torch
from typing import List
from datetime import datetime
from lightning import LightningModule
from india_benchmark.utils.forecast_metrics import lat_weighted_mse, lat_weighted_rmse


class ClimatologyModule(LightningModule):
    def __init__(self, variables: List[str], vars_to_log: List[str], climatology_path: str):
        super().__init__()
        self.climatology = xr.open_zarr(climatology_path)
        self.save_hyperparameters()
    
    def extract_time_from_filename(self, filename):
        basename = os.path.basename(filename)
        date_str, time_str = basename.split('_')
        time_str = time_str.split('.')[0]
        
        time_map = {
            '00': '00:00:00',
            '01': '06:00:00',
            '02': '12:00:00',
            '03': '18:00:00'
        }
        time_of_day = time_map[time_str]
        
        return f"{date_str}T{time_of_day}"
    
    def extract_clim_to_numpy(self, ds, variables_list):
        T = len(ds.time)
        H = len(ds.latitude)
        W = len(ds.longitude)
        C = len(variables_list)
        
        pressure_levels = {
            '925': 925.0,
            '850': 850.0,
            '700': 700.0,
            '600': 600.0,
            '500': 500.0,
            '250': 250.0,
            '50': 50.0
        }
        
        # Initialize output array
        output = np.zeros((T, C, H, W), dtype=np.float32)
        
        for c, var_name in enumerate(variables_list):            
            if any(suffix in var_name for suffix in pressure_levels.keys()):
                # For variables like HGT925, TMP_prl850, etc.
                # Find the base variable name and pressure level
                for suffix, pres_value in pressure_levels.items():
                    if var_name.endswith(suffix):
                        level = pres_value
                        base_var = var_name[:-len(suffix)]
                        break
                
                # Select the variable at the specific pressure level
                data = ds[base_var].sel(isobaricInhPa=level).values
                
                output[:, c, :, :] = data
            else:
                # For variables without pressure level specification (e.g., TMP, UGRD)
                output[:, c, :, :] = ds[var_name].values
        
        return output
        
    def get_clim_predictions(self, batch_filenames):
        """
        filenames: list of B sublists, each sublist contains pred_steps filenames
        """
        b = len(batch_filenames)
        predictions = []
        for i in range(b):
            filenames = batch_filenames[i]
            datetime_stres = [self.extract_time_from_filename(f) for f in filenames]
            clim = self.climatology.sel(time=datetime_stres)
            # extract relevant variables
            pred = self.extract_clim_to_numpy(clim, self.hparams.variables)
            predictions.append(pred)
        return np.array(predictions)

    def evaluate_step(self, batch, batch_idx, split):
        init_states, true_states, filenames = batch
        init_steps = init_states.shape[1]
        target_filenames = [f[init_steps:] for f in filenames]
        prediction = self.get_clim_predictions(target_filenames)  # (B, pred_steps, C, H, W)
        prediction = torch.from_numpy(prediction).to(true_states.device, dtype=true_states.dtype)
        variables = self.hparams.variables
        
        loss_dict = {}
                
        for step in range(true_states.shape[1]):
            denormalized_pred = prediction[:, step:step+1] # not normalized in the first place
            denormalized_target = self.trainer.datamodule.denormalize(true_states[:, step:step+1])
            lead_time = self.trainer.datamodule.hparams.lead_time * (step + 1)
            wrmse_loss_dict = lat_weighted_rmse(
                denormalized_pred, denormalized_target,
                vars=variables,
                lat=self.trainer.datamodule.lat,
                chosen_vars=self.hparams.vars_to_log or variables,
                postfix=f"_{lead_time:03d}h",
            )
            loss_dict.update(wrmse_loss_dict)
        
        loss_dict = {f"{split}/{k}": v for k, v in loss_dict.items()}
        
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
    
    def validation_step(self, batch, batch_idx):
        self.evaluate_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        self.evaluate_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        return None