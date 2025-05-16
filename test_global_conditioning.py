import os
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from india_benchmark.datasets.coupling_datamodule import CouplingDataModule
from india_benchmark.models.global_conditioning_module import GlobalConditioningModule


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_path", type=str, required=True)

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = CustomCLI(
        model_class=GlobalConditioningModule,
        datamodule_class=CouplingDataModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    logger_name = cli.trainer.logger._name
    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints'),
                monitor=cli.trainer.callbacks[i].monitor,
                mode=cli.trainer.callbacks[i].mode,
                save_top_k=cli.trainer.callbacks[i].save_top_k,
                save_last=cli.trainer.callbacks[i].save_last,
                verbose=cli.trainer.callbacks[i].verbose,
                filename=cli.trainer.callbacks[i].filename,
                auto_insert_metric_name=cli.trainer.callbacks[i].auto_insert_metric_name
            )

    cli.trainer.logger = WandbLogger(
        name=logger_name,
        project=cli.trainer.logger._wandb_init['project'],
        save_dir=os.path.join(cli.trainer.default_root_dir, logger_name)
    )
    os.makedirs(os.path.join(cli.trainer.default_root_dir, logger_name, 'wandb'), exist_ok=True)

    assert os.path.exists(cli.config.ckpt_path), f"Checkpoint file {cli.config.ckpt_path} does not exist."

    # fit() runs the training
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)


if __name__ == "__main__":
    main()
