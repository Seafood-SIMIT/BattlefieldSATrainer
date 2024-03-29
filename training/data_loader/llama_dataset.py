from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from typing import Optional

def pad(ids, pad_id, max_length):
    if len(ids) > max_length:
        return ids[:max_length]
    return ids + [pad_id] * (max_length - len(ids))

class LlamaDataset(Dataset):
    '''
    Dataset Used for yuyuan medical qa task.
    Just surpport small datasets, when deal with large datasets it may be slowly.
    for large datasets please use mmapdatasets(doing)
    '''

    def __init__(self, args,tokenizer,data_set,add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        #self.data_size = os.path.getsize(args.data_path)/1024/1024/1024
        self.data = data_set
        self.max_seq_length = args.max_seq_length
        #self.max_seq_length =-1
        self.add_special_tokens = add_special_tokens
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index])

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
        r_qb = item['prompt'][0]

        r_sa = item['output'][0]
        
        prompt_ids = self.tokenizer(r_qb, add_special_tokens=False)['input_ids']

        sa_ids = self.tokenizer(r_sa, add_special_tokens=False)['input_ids']
        
        labels_ids = [-100] * (len(prompt_ids)) + sa_ids

        input_ids = pad(prompt_ids, self.tokenizer.eos_token_id, self.max_seq_length)
        target_ids = pad(labels_ids, -100, self.max_seq_length)

        return  {"input_ids": torch.tensor(input_ids).clone().detach(),
                 "attention_mask": torch.ones( self.max_seq_length).clone().detach(), 
                 "position_ids": torch.arange(0, self.max_seq_length).clone().detach(),
                 "labels":  torch.tensor(target_ids).clone().detach()}
    
class WYLLamaDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, args_data):
        super().__init__()
        if args_data.hf_data:
            self.datasets = load_dataset(args_data.hf_data)
        else:
            self.datasets = load_dataset(args_data.raw_file_type, data_dir = args_data.data_dir, data_files={'train':'train.json',
                                                                                                        'validation':'valid.json',
                                                                                                        })
        self.train_dataset = LlamaDataset(args_data,tokenizer,self.datasets['train'])
        self.valid_dataset = LlamaDataset(args_data,tokenizer,self.datasets['validation'])
        
        self.args_data = args_data
        self.tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        return 

    def train_dataloader(self):
        ds = self.train_dataset

        return DataLoader(
            ds,
            shuffle=True,
            batch_size=self.args_data.train_batchsize,
            num_workers=self.args_data.num_workers,
        )

    def val_dataloader(self):
        ds = self.valid_dataset
        return DataLoader(
            ds,
            batch_size=self.args_data.valid_batchsize,
            shuffle=False,
            num_workers=self.args_data.num_workers,
            pin_memory=False,
        )

        # return DataLoader(
        #     ds, shuffle=False, batch_size=self.hparams.val_batchsize, pin_memory=False, collate_fn=collate_fn,
        # )

    def test_dataloader(self):
        ds = self.valid_dataset

        return DataLoader(
            ds,
            batch_size=self.args_data.val_batchsize,
            shuffle=False,
            num_workers=self.args_data.num_workers,
            pin_memory=False,
        )