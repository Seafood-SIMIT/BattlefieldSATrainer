"""Experiment-running framework."""
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
import torch

import os
os.chdir('/home/seafood/wkdir/kg_trainer')

import sys
sys.path.append('.')

from utils import HParam
from kg_generator import kg_model_transformer_generator,kg_model_bert_generator, KG_BaseLitModel
from gpt2_generator import gpt2_model_gpt2_generator, GPT2_BaseLitModel
from training.util import import_class, setup_data_from_args


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


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

    #kg_model = kg_model_transformer_generator(hp.kg.pretrained_file)
    kg_model = kg_model_bert_generator(hp.kg.pretrained_file)
    gpt2_model = gpt2_model_gpt2_generator(hp.gpt2.pretrained_file)
    #data
    data= setup_data_from_args(hp)

    kg_litmodel = KG_BaseLitModel
    gpt2_litmodel = GPT2_BaseLitModel

    if hp.kg.load_checkpoint != 'None':
        kg_litmodel = kg_litmodel.load_from_checkpoint(hp.litmodel.load_checkpoint, args=hp, model=kg_model)
    else:
        lit_model = kg_litmodel(args=hp.kg, model=kg_model)

    if hp.gpt2.load_checkpoint != 'None':
        gpt2_litmodel = gpt2_litmodel.load_from_checkpoint(hp.litmodel.load_checkpoint, args=hp, model=gpt2_model)
    else:
        gpt2_litmodel = gpt2_litmodel(args=hp.gpt2, model=gpt2_model)

    # Call baks
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
    if hp.train.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation/loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    trainer = pl.Trainer(devices=[5,6],accelerator='gpu',
                        logger = logger)

    #trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(kg_litmodel, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
