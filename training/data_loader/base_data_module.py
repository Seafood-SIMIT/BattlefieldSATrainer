"""Base DataModule class."""
import argparse
import os
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from gpt2_generator import util
from .util import BaseDataset


def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename


BATCH_SIZE = 128
#NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()

# sensible multiprocessing defaults: at most one worker per CPU
#DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
DEFAULT_NUM_WORKERS = 8
# but in distributed data parallel mode, we launch a training on each GPU, so must divide out to keep total at one worker per CPU
#DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS



class BaseDataModule(pl.LightningDataModule):
    """Base for all of our LightningDataModules.

    Learn more at about LDMs at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.hp = args
        self.batch_size = self.hp.data.batch_size
        self.num_workers = self.hp.data.num_workers

        #self.on_gpu = isinstance(self.hp.data.gpus, (str, int))

        # Make sure to set the variables below in subclasses
        self.train_dataset: Union[BaseDataset, ConcatDataset]
        self.val_dataset: Union[BaseDataset, ConcatDataset]
        self.test_dataset: Union[BaseDataset, ConcatDataset]

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""

    def prepare_data(self, *args, **kwargs) -> None:
        """Take the first steps to prepare data for use.

        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.

        Here is where we typically split into train, validation, and test. This is done once per GPU in a DDP setting.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
