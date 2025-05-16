from typing import List
from lightning import LightningModule
from india_benchmark.utils.forecast_metrics import lat_weighted_mse, lat_weighted_rmse

class PersistenceModule(LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """
    def __init__(self, variables: List[str], vars_to_log: List[str]):
        super().__init__()
        self.save_hyperparameters()

    def unroll_prediction(self, init_states, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, H, C, H, W)
        true_states: (B, pred_steps, C, H, W)
        """
        pred_steps = true_states.shape[1]        
        predictions = init_states[:, -1]
        predictions = predictions.unsqueeze(1).repeat(1, pred_steps, 1, 1, 1)
        return predictions

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
        return None