# coding=utf8
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import csv
import json

class GPT2QADataset(Dataset):
    '''
    Dataset Used for yuyuan medical qa task.
    Just surpport small datasets, when deal with large datasets it may be slowly.
    for large datasets please use mmapdatasets(doing)
    '''

    def __init__(self, args,data_path,tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        #self.data_size = os.path.getsize(args.data_path)/1024/1024/1024
        self.data_type_name = args.data_type_name
        self.data = self.load_data(data_path)
        self.max_seq_length = args.max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index])

    def load_data(self, data_path):
        # 有进度条展示
        #if self.data_size <= 5:
        with tqdm(total=None, desc=f'{self.data_type_name}处理进度', mininterval=0.3) as bar:
            data = []
            for a_file in os.listdir(data_path):
                # for mac osx
                if a_file.startswith('.'):
                    continue

                
                with open(os.path.join(data_path,a_file), "r", encoding='utf8') as f:
                    lines = json.load(f)
                    for line in lines:
                        data.append(line)
                    bar.update()

        #if self.data_size > 5:
        #    data_gen.close()
        return data

    def data_parse(self, line):
        """
        解析不同格式的数据
        """
        #dic = csv.reader(line,delimiter=',')
        return line

    def encode(self, item):
        """
        将数据转换成模型训练的输入
        """
        # time, mid, blue, red,trible, discription
        # conver to ids
        r_qb = item['时刻']+';'+item['中立情报']+';'+item['获取情报']+';'+item['我方情报']
        r_sa = item['态势描述']
        inputs_dict = self.tokenizer.encode_plus(r_qb+r_sa,
                                                 max_length=self.max_seq_length, padding='max_length',
                                                 truncation=True, return_tensors='pt')
        target = inputs_dict['input_ids']
        labels = target.clone().detach()
        labels[target == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs_dict['input_ids'].squeeze(),
            "attention_mask": inputs_dict['attention_mask'].squeeze(),
            "labels": labels.squeeze(),
            "question": r_qb,
            "answer": r_sa
        }


class WenzhongQADataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('GPT2QADataModel')
        parser.add_argument('--data_dir', type=str, required=True)
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--train_data', default='train.txt', type=str)
        parser.add_argument('--valid_data', default='valid.txt', type=str)
        parser.add_argument('--test_data', default='test.txt', type=str)
        parser.add_argument('--train_batchsize', type=int, required=True)
        parser.add_argument('--valid_batchsize', type=int, required=True)
        parser.add_argument('--max_seq_length', default=1024, type=int)
        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        if not args.do_eval_only:
            self.train_data = GPT2QADataset(args,os.path.join(args.data_path,'train'),tokenizer)
            self.valid_data = GPT2QADataset(args,os.path.join(args.data_path,'valid'),tokenizer)
        self.test_data = self.valid_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data, shuffle=True,
            batch_size=self.train_batchsize,
            pin_memory=False, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False,
                          batch_size=self.valid_batchsize,
                          pin_memory=False, num_workers=self.args.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False,
                          batch_size=self.valid_batchsize, pin_memory=False,
                          num_workers=self.args.num_workers)


if __name__ == '__main__':
    import argparse
    datafile = 'files/dataset/demo/raw/demo.csv'

    testml = GPT2QADataset(datafile, 'medical_qa', args=hp.data)

    print(testml[10])