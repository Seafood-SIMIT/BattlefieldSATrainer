from .base_data_module import BaseDataModule
from .dataset_wyl import readData,WYLDataset

import numpy as np
class WHOLEYULIAO(BaseDataModule):
    """WHOLE yuliao"""

    def __init__(self, hp, tokenizer) -> None:
        super().__init__(hp)
        """Data dir info"""
        self.hp = hp
        self.tokenizer = tokenizer

    def prepare_data(self) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        data= readData(self.hp, self.tokenizer)
        print('demo data',data[len(data)//2])
        #data_train, data_val = random_split(samples, [10, 7])  # type: ignore
        self.train_dataset = WYLDataset(data)
        #self.val_dataset = PYLDataset(data_set=np.array(data_val))
        self.val_dataset = self.train_dataset 
        
        #split the dataset
        #print(self.train_dataset.shape)
        self.test_dataset = self.val_dataset


if __name__ == "__main__":
    load_and_print_info(PUREYULIAO)