from typing import List
import torch
from lightning import LightningModule
from india_benchmark.utils.forecast_metrics import lat_weighted_mse, lat_weighted_rmse
from india_benchmark.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

class BoundaryForcingModule(LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """
    def __init__(
        self,
        net: torch.nn.Module,
        img_size: List[int],
        boundary_pixels: int,
        variables: List[str],
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

        boundary_mask = self.get_boundary_mask(img_size, boundary_pixels) # (h, w)
        self.register_buffer("boundary_mask", boundary_mask, persistent=False) # 1 for border
        self.register_buffer("interior_mask", 1.0 - boundary_mask, persistent=False)  # 1 for non-border
    
    def get_boundary_mask(self, img_size, boundary_pixels):
        # 1 for border, 0 for interior
        h, w = img_size
        mask = torch.zeros(h, w)
        mask[:boundary_pixels, :] = 1
        mask[-boundary_pixels:, :] = 1
        mask[:, :boundary_pixels] = 1
        mask[:, -boundary_pixels:] = 1
        return mask

    def predict_step(self, prev_states):
        """
        Step state one step ahead using prediction model, X_{t-H:t} -> X_t+1
        prev_states: (B, H, C, H, W)
        """
        pred_delta_mean = self.net(prev_states)  # (B, C, H, W)
        # Rescale with one-step difference statistics
        rescaled_delta_mean = self.trainer.datamodule.denormalize_diff(pred_delta_mean)
        rescaled_prev_state = self.trainer.datamodule.denormalize(prev_states[:, -1])
        new_state = rescaled_delta_mean + rescaled_prev_state
        new_state = self.trainer.datamodule.normalize(new_state)
        return new_state

    def unroll_prediction(self, init_states, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, H, C, H, W)
        true_states: (B, pred_steps, C, H, W)
        """
        pred_steps = true_states.shape[1]
        prediction_list = []

        for i in range(pred_steps):
            pred_state = self.predict_step(init_states) # (B, C, H, W)
            # Overwrite border with true state
            border_state = true_states[:, i] # (B, C, H, W)
            new_state = self.boundary_mask * border_state + self.interior_mask * pred_state

            prediction_list.append(new_state)

            # Update conditioning states
            init_states = torch.cat([init_states[:, 1:], new_state.unsqueeze(1)], dim=1)

        prediction = torch.stack(prediction_list, dim=1)  # (B, pred_steps, C, H, W)

        return prediction

    def training_step(self, batch, batch_idx):
        """
        Train on single batch
        """
        init_states, true_states = batch
        prediction = self.unroll_prediction(init_states, true_states)  # (B, pred_steps, C, H, W)

        # Compute loss
        wmse_loss_dict = lat_weighted_mse(
            prediction, true_states,
            vars=self.hparams.variables,
            lat=self.trainer.datamodule.lat
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
        init_states, true_states = batch
        prediction = self.unroll_prediction(init_states, true_states)  # (B, pred_steps, C, H, W)
        variables = self.hparams.variables

        # Compute loss
        wmse_loss_dict = lat_weighted_mse(
            prediction, true_states,
            vars=variables,
            lat=self.trainer.datamodule.lat
        )
        wmse_loss = wmse_loss_dict["w_mse_agg"]
        loss_dict = {
            f"w_mse_agg": wmse_loss.item()
        }
        
        for step in range(true_states.shape[1]):
            denormalized_pred = self.trainer.datamodule.denormalize(prediction[:, step:step+1])
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