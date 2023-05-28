"""Experiment-running framework."""
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
import torch

import os
os.chdir('kg_trainer')

from utils import HParam
from gpt2_generator import lit_models
from training.util import import_class, setup_data_and_model_from_args


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    return args, hp


@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)


    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("-c",'--config',default='config/default.yaml', type=str, help='set the config file')
    parser.add_argument("-m", type=str, required= True, help='model name')

    args = parser.parse_args()
    hp = HParam(args.config)


    #data
    data= setup_data_and_model_from_args(hp)

    lit_model_class = lit_models.GPT2Chinese

    if hp.litmodel.load_checkpoint != 'None':
        lit_model = lit_model_class.load_from_checkpoint(hp.litmodel.load_checkpoint, args=hp, model=model)
    else:
        lit_model = lit_model_class(args=hp.litmodel)

    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/cer" if hp.litmodel.loss in ("transformer",) else "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=hp.chkpt.every_n_epochs,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]
    if hp.litmodel.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation/loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    trainer = pl.Trainer(devices=[0,],accelerator='cuda',logger = logger)
    #trainer = pl.Trainer(devices=1,accelerator='gpu', max_epochs=5)

    #trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
