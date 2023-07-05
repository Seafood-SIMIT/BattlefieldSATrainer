from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional

@dataclass
class WYLCOllator:
    tokenizer: None
    #args_data: None
    def __call__(self,samples):
        input_ids_list = []
        labels_list = []
        max_length = 0
        for item in samples:
            """
            Samples: ['时刻', '获取情报', '中立情报', '我方情报', '态势描述'],
            """
            r_qb = item['时刻']+';'+item['中立情报']+';'+item['获取情报']+';'+item['我方情报']+';'
            r_sa = item['态势描述']

            prompt_input_ids = self.tokenizer(r_qb, add_spectial_tokens=False).input_ids
            output_ids = self.tokenizer(r_sa, add_spectial_tokens=False).input_ids+[self.tokenizer.eos_token_id]

            max_length = min(max(len(prompt_input_ids), max_length), self.max_seq_length)
            input_ids_list.append(prompt_input_ids)
            labels_list.append(output_ids)

        for i in range(len(input_ids_list)):
            labels_list[i] = self.pad(labels_list[i],-100,max_length)
            input_ids_list[i] = self.pad(input_ids_list[i], self.tokenizer.eos_token_id, max_length)
        model_inputs = {
            'input_ids': torch.tensor(input_ids_list).clone(),
            'attention_mask': torch.ones((len(input_ids_list), max_length)).clone(),
            "position_ids": torch.arange(0, max_length).unsqueeze(0).expand(len(input_ids_list), max_length).clone(),
            'labels': torch.tensor(labels_list).clone(),
        }
        return model_inputs

    def pad(ids, pad_id, max_length):
        if len(ids) > max_length:
            return ids[:max_length]
        return ids + [pad_id] * (max_length - len(ids))
    
class WYLLamaDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, collate_fn, args_data):
        super().__init__()
        self.train_dataset = load_dataset(args_data.raw_file_type, data_dir = args_data.data_dir, data_files='train.json')
        self.valid_dataset = load_dataset(args_data.raw_file_type, data_dir = args_data.data_dir, data_files='valid.json')

        self.tokenizer = tokenizer
        self.collate_fn  = collate_fn

    def setup(self, stage: Optional[str] = None) -> None:
        return

    def train_dataloader(self):
        ds = self.train_datasets

        collate_fn = self.collate_fn
        if hasattr(ds, 'collate_fn'):
            collate_fn = ds.collate_fn

        return DataLoader(
            ds,
            batch_size=self.args_data.train_batchsize,
            num_workers=self.args_data.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        ds = self.valid_datasets
        collate_fn = self.collate_fn
        if hasattr(ds, 'collate_fn'):
            collate_fn = ds.collate_fn

        return DataLoader(
            ds,
            batch_size=self.args_data.valid_batchsize,
            shuffle=False,
            num_workers=self.args_data.num_workers,
            collate_fn=collate_fn,
        )

        # return DataLoader(
        #     ds, shuffle=False, batch_size=self.hparams.val_batchsize, pin_memory=False, collate_fn=collate_fn,
        # )

    def test_dataloader(self):
        ds = self.valid_datasets
        collate_fn = self.collate_fn
        if hasattr(ds, 'collate_fn'):
            collate_fn = ds.collate_fn

        return DataLoader(
            ds,
            batch_size=self.args_data.val_batchsize,
            shuffle=False,
            num_workers=self.args_data.num_workers,
            collate_fn=collate_fn,
        )