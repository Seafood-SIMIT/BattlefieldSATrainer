"""Experiment-running framework."""
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch

import os
#os.chdir('/home/seafood/wkdir/kg_trainer')

import sys
sys.path.append('.')

from utils import HParam
from kg_generator import *
from gpt2_generator import gpt2_model_gpt2_generator, GPT2_BaseLitModel, WenzhongQALitModel
from training.util import import_class, setup_data_from_args

#import nemo
#from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

#from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


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
    parser.add_argument("-m","--model_name", type=str, required= True, help='model name')

    args = parser.parse_args()
    hp = HParam(args.config)

    gpt2_model = gpt2_model_gpt2_generator(hp.gpt2.pretrained_file)
    #data
    data, tokenizer= setup_data_from_args(hp)

    gpt2_litmodel = WenzhongQALitModel


    if hp.gpt2.load_checkpoint :
        gpt2_litmodel = gpt2_litmodel.load_from_checkpoint(hp.litmodel.load_checkpoint, args=hp, model=gpt2_model)
    else:
        gpt2_litmodel = gpt2_litmodel(args=hp.gpt2, model=gpt2_model,num_data=len(data.train_dataloader()))

    # Call baks
    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.WandbLogger(project='BASAer',name=args.model_name,save_dir=log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/cer" if hp.gpt2.loss in ("transformer",) else "val_loss"
    filename_format = "epoch={epoch:04d}-validation.loss={val_loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=hp.trainer.chkpt_every_n_epochs,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]
    if hp.trainer.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    trainer = pl.Trainer(devices=hp.trainer.devices,accelerator=hp.trainer.accelerator,
                        max_epochs=hp.trainer.max_epochs,
                        strategy=hp.trainer.strategy,
                        #strategy=NLPDDPStrategy(),
                        callbacks=callbacks,
                        logger = logger)

    #trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(gpt2_litmodel, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(gpt2_litmodel, datamodule=data)


if __name__ == "__main__":
    main()
