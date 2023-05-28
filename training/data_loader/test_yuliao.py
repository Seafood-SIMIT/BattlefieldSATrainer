import numpy as np
import argparse
from .base_data_module import BaseDataModule, load_and_print_info
import os
from tqdm import tqdm
from torch.utils.data import random_split
from .dataset_test import KGDataset
import pytorch_lightning as pl
import jieba
import torch
N_CTX = 1024
TRAIN_SIZE = 0.7
VAL_SIZE = 0.3
DATA_RAW = True
STRIDE=678



def build_files(tokenizer):
    #
    # total dataset
    #open file
    raw_qb = ['红方载具击伤蓝方直升机，红方步兵多名由D6绕至D7','红方载具击伤蓝方直升机，红方步兵多名由D6绕至D7']
    raw_sa = [[{"entity1":"红方载具","relation":"击伤","entity2":"蓝方直升机"},
            {'entity1':"红方步兵","relation":"至","entity2":"D7"},
            {'entity':"红方步兵","shuxing":"多名"}],[{"entity1":"红方载具","relation":"击伤","entity2":"蓝方直升机"},
            {'entity1':"红方步兵","relation":"至","entity2":"D7"},
            {'entity':"红方步兵","shuxing":"多名"}]]
    full_data=[]
    for text in raw_qb:
        data={}

        jieba.load_userdict("files/tokenizer/jieba_dict.txt")
        words = list(jieba.cut(text))
        for word in words:
            if word not in tokenizer.vocab:
                tokenizer.add_tokens([word])

        tokenizer.save_vocabulary('files/tokenizer/vocab_2.txt')
        
        inputs = tokenizer.encode_plus(words,
                                    add_special_tokens=True,
                                        return_attention_mask=True,
                                        return_tensors="pt")

        #offsets = encoding['offset_mapping']

        label = np.zeros(len(inputs['attention_mask'][0]))
        for sa in raw_sa[raw_qb.index(text)]:
            if len(sa) == 3:
                entity = sa['entity1']
                label[text.index(entity):text.index(entity)+len(entity)] = 1
                entity = sa['entity2']
                label[text.index(entity):text.index(entity)+len(entity)] = 1
                relation = sa['relation']
                label[text.index(relation):text.index(relation)+len(relation)] = 2
            elif len(sa) == 2:
                entity = sa['entity']
                label[text.index(entity):text.index(entity)+len(entity)] = 1
                shuxing = sa['shuxing']
                label[text.index(shuxing):text.index(shuxing)+len(shuxing)] = 3
        data['text']=text
        data['input_ids']=inputs['input_ids']
        data['attention_mask'] = inputs['attention_mask']
        data['label']=torch.tensor(label,dtype=torch.long)
        full_data.append(data)

    return full_data

def readData(tokenizer):
    data= build_files(tokenizer)

        #先这样用，采取n_ctx 和 stride
    start_point = 0
    return data

class KGYULIAO(BaseDataModule):
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
        data = readData(self.tokenizer)
        print('test data example',data)
        #data_train, data_val = random_split(samples, [10, 7])  # type: ignore
        self.train_dataset = KGDataset(data=data)
        #self.val_dataset = PYLDataset(data_set=np.array(data_val))
        self.val_dataset = self.train_dataset 
        
        #split the dataset
        #print(self.train_dataset.shape)
        self.test_dataset = self.val_dataset


if __name__ == "__main__":
    load_and_print_info(PUREYULIAO)



