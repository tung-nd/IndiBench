from typing import List, Union
import torch
from lightning import LightningModule
from india_benchmark.models.networks.stormcast_unet import StormCastUNet
from india_benchmark.models.networks.unet import Unet
from india_benchmark.models.networks.stormer import Stormer
from india_benchmark.models.neural_lam.models.hi_lam import HiLAM
from india_benchmark.models.neural_lam.models.graph_lam import GraphLAM
from india_benchmark.utils.forecast_metrics import lat_weighted_mse, lat_weighted_rmse
from india_benchmark.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

class GlobalConditioningModule(LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """
    def __init__(
        self,
        net: Union[StormCastUNet, Unet, Stormer, HiLAM, GraphLAM],
        vars_to_log: List[str],
        lr: float,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

    def predict_step(self, prev_states, local_channels):
        """
        Step state one step ahead using prediction model, X_{t-H:t} -> X_t+1
        prev_states: (B, C, H, W)
        """
        pred_delta_mean = self.net(prev_states)  # (B, C, H, W)
        # Rescale with one-step difference statistics
        rescaled_delta_mean = self.trainer.datamodule.denormalize(pred_delta_mean, 'diff', 'local')
        rescaled_prev_state = self.trainer.datamodule.denormalize(prev_states[:, -1, :local_channels], 'state', 'local')
        new_state = rescaled_delta_mean + rescaled_prev_state
        new_state = self.trainer.datamodule.normalize(new_state, 'state', 'local')
        return new_state

    def unroll_prediction(self, local_inp, global_inp, global_out):
        """
        Roll out prediction taking multiple autoregressive steps with model
        local_inp: (B, inp_steps, C1, H, W)
        global_inp: (B, inp_steps, C2, H, W)
        global_out: (B, pred_steps, C2, H, W)
        """
        pred_steps = global_out.shape[1]
        local_channels = local_inp.shape[2]
        prediction_list = []
        for i in range(pred_steps):
            inp = torch.cat([local_inp, global_inp], dim=2)  # (B, inp_steps, C1 + C2, H, W)
            
            local_pred = self.predict_step(inp, local_channels)  # (B, C1, H, W)
            global_pred = global_out[:, i]  # (B, C2, H, W), in an operational setting this would be prediction from a global model
            prediction_list.append(local_pred)
            
            # update local_inp and global_inp
            local_inp = torch.cat([
                local_inp[:, 1:], local_pred.unsqueeze(1)
            ], dim=1)
            global_inp = torch.cat([
                global_inp[:, 1:], global_pred.unsqueeze(1)
            ], dim=1)

        prediction = torch.stack(prediction_list, dim=1)  # (B, pred_steps, C1, H1, W1)

        return prediction

    def training_step(self, batch, batch_idx):
        """
        Train on single batch
        """
        global_inp, local_inp, global_out, local_out = batch
        prediction = self.unroll_prediction(local_inp, global_inp, global_out)  # (B, pred_steps, C, H, W)

        # Compute loss
        wmse_loss_dict = lat_weighted_mse(
            prediction, local_out,
            vars=self.trainer.datamodule.hparams.local_variables,
            lat=self.trainer.datamodule.local_lat
        )
        loss = wmse_loss_dict["w_mse_agg"]

        self.log(
            "train/w_mse_agg",
            loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def evaluate_step(self, batch, batch_idx, split):
        global_inp, local_inp, global_out, local_out = batch
        prediction = self.unroll_prediction(local_inp, global_inp, global_out)  # (B, pred_steps, C, H, W)
        variables = self.trainer.datamodule.hparams.local_variables

        # Compute loss
        wmse_loss_dict = lat_weighted_mse(
            prediction, local_out,
            vars=variables,
            lat=self.trainer.datamodule.local_lat
        )
        wmse_loss = wmse_loss_dict["w_mse_agg"]
        loss_dict = {
            f"w_mse_agg": wmse_loss.item()
        }
        
        for step in range(local_out.shape[1]):
            denormalized_pred = self.trainer.datamodule.denormalize(prediction[:, step:step+1], type="state", scope="local")
            denormalized_target = self.trainer.datamodule.denormalize(local_out[:, step:step+1], type="state", scope="local")
            lead_time = self.trainer.datamodule.hparams.lead_time * (step + 1)
            wrmse_loss_dict = lat_weighted_rmse(
                denormalized_pred, denormalized_target,
                vars=variables,
                lat=self.trainer.datamodule.local_lat,
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            weight_decay=1e-5
        )

        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,
            self.hparams.max_epochs * n_steps,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}