import numpy as np
import argparse
from .base_data_module import BaseDataModule, load_and_print_info
import os
from tqdm import tqdm
from torch.utils.data import random_split
from .dataset_pyl import PYLDataset, readData

N_CTX = 1024
TRAIN_SIZE = 0.7
VAL_SIZE = 0.3
DATA_RAW = True
STRIDE=678

class PUREYULIAO(BaseDataModule):
    """PURE yuliao"""

    def __init__(self, hp, tokenizer) -> None:
        super().__init__(hp)
        """Data dir info"""
        self.hp = hp
        self.tokenizer = tokenizer

    def prepare_data(self) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        samples = readData(self.hp, self.tokenizer)
        data_train=samples
        #data_train, data_val = random_split(samples, [10, 7])  # type: ignore
        self.train_dataset = PYLDataset(data_set=np.array((data_train)))
        #self.val_dataset = PYLDataset(data_set=np.array(data_val))
        self.val_dataset = self.train_dataset 
        
        #split the dataset
        #print(self.train_dataset.shape)
        self.test_dataset = self.val_dataset


if __name__ == "__main__":
    load_and_print_info(PUREYULIAO)



