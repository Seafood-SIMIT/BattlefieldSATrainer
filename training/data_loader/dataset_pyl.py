import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

import os
from tqdm import tqdm
import json


def build_files(tokenized_data_path, full_tokenizer, raw_path):
    if not os.path.exists(os.path.join(tokenized_data_path)):
        os.mkdir(os.path.join(tokenized_data_path))
    
    #
    file_list = os.listdir(raw_path)
    for i in tqdm(file_list):
        if i.startswith('.'):
            continue
        with open(os.path.join(raw_path, i),'r') as f:
            sublines = f.read().strip()
        sublines = sublines.split()
        sublines = [full_tokenizer.tokenize(line) for line in sublines]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(os.path.join(tokenized_data_path,'tokenized_train_{}.txt'.format(i)), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')


def readData(hp, full_tokenizer):
    if hp.data.raw:
        build_files(hp.data.train_dir, full_tokenizer = full_tokenizer, raw_path = hp.data.raw_dir)
    samples = []
    
    train_list = os.listdir(hp.data.train_dir)
    for a_file in train_list:
        if a_file.startswith('.'):
            continue
        with open(os.path.join(hp.data.train_dir, a_file), 'r') as f:
            line = f.read().strip()

        tokens = line.split()
        tokens = [int(token) for token in tokens]

        #先这样用，采取n_ctx 和 stride
        start_point = 0
        while start_point < len(tokens) - hp.data.n_ctx:
            samples.append(tokens[start_point: start_point + hp.data.n_ctx])
            start_point += hp.data.stride
        if start_point < len(tokens):
            samples.append(tokens[len(tokens)-hp.data.n_ctx:])

    return np.array(samples)
    #build files

    
class PYLDataset(Dataset):
    def __init__(self, data_set):
        self.sequence = data_set[0:2]
        self.label = data_set
        self.tensor_sequence = torch.tensor(self.sequence, dtype=torch.long)
        self.tensor_label = torch.tensor(self.label, dtype=torch.long)
        
        ##划分测试集
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.tensor_label[idx], self.tensor_label[idx]
        #return 0, 0

if __name__ == "__main__":
    a = createDataloader()