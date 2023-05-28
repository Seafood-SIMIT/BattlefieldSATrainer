import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

import os
from tqdm import tqdm
import json
import csv


def build_files(tokenized_data_path, tokenizer, raw_path):
    #
    file_list = os.listdir(raw_path)
    # total dataset
    for i in tqdm(file_list):
        #delete osx files
        if i.startswith('.'):
            continue

        full_data=[]
        #open file
        max_qb_length, max_sa_length=0,0
        with open(os.path.join(raw_path, i),'r') as f:
            sublines = csv.reader(f,delimiter=',')
            for line in sublines:
                data={}
                # time, mid, blue, red,trible, discription
                r_time, r_mid_qb, r_blue_qb, _,r_red_qb,_, r_sa = line
                # conver to ids
                r_qb = r_time+';'+r_mid_qb+';'+r_blue_qb+';'+r_red_qb
                inputs = tokenizer.encode_plus(r_qb,
                                        max_length = 1024,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        return_attention_mask=True,
                                        return_tensors="pt")
                sas = tokenizer.encode_plus(r_sa,
                                        max_length = 1024,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        return_attention_mask=False,
                                        return_tensors="pt")
                data['text'] = r_qb
                data['input_ids'] = inputs['input_ids']
                data['attention_mask'] = inputs['attention_mask']
                data['label'] = sas['input_ids']
                full_data.append(data)

                #add to list
    print('read CSV finish')
    return full_data


def readData(hp, full_tokenizer):
    data = build_files(hp.data.train_dir, tokenizer = full_tokenizer, raw_path = hp.data.raw_dir)

        #先这样用，采取n_ctx 和 stride

    # design the training method
    return data
   
   
    #build files

    
class WYLDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
        ##划分测试集
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids'].squeeze(0)
        attention_mask = self.data[idx]['attention_mask'].squeeze(0)
        label_ids = self.data[idx]['label'].squeeze(0)
        return input_ids, attention_mask, label_ids
        #return 0, 0

if __name__ == "__main__":
    a = createDataloader()