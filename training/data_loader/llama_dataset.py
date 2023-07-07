from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional

def pad(ids, pad_id, max_length):
    if len(ids) > max_length:
        return ids[:max_length]
    return ids + [pad_id] * (max_length - len(ids))


prompt_prefix = ""
prompt_without_output = "<human>:{prompt}\n<bot>:"

@dataclass
class WYLCOllator:
    '''
    由input处理成samples，也就是最终模型的输入
    其中主要处理逻辑在__call__里
    '''
    tokenizer: None  # 分词
    max_seq_length: 1536
    def __call__(self, samples):
        input_ids_list = []
        labels_list = []
        max_length = 0
        for s in samples:
            """
            sample: {
                "task" : str,
                "prompt": [str]
                "output": [str]
                }
            """
            prompt_cnt = min(len(s["prompt"]), len(s["output"]))
            # input_ids = self.tokenizer(prompt_prefix).input_ids
            input_ids = []
            labels_ids = [-100] * len(input_ids)
            for i in range(prompt_cnt):
                prompt_input_ids = self.tokenizer(prompt_without_output.format_map(
                    {"prompt": s["prompt"][i].strip()}), add_special_tokens=False).input_ids
                output_ids = self.tokenizer(s["output"][i].strip(), add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                
                input_ids += prompt_input_ids
                input_ids += output_ids
                
                labels_ids += [-100] * (len(prompt_input_ids)) + output_ids
            
            # input_ids += [self.tokenizer.eos_token_id]
            # labels_ids += [self.tokenizer.eos_token_id]
            max_length = min(max(len(input_ids), max_length), self.max_seq_length)
            input_ids_list.append(input_ids)
            labels_list.append(labels_ids)

        # PAD
        for i in range(len(input_ids_list)):
            labels_list[i] = pad(labels_list[i], -100, max_length)
            input_ids_list[i] = pad(input_ids_list[i], self.tokenizer.eos_token_id, max_length)
        model_inputs = {
            'input_ids': torch.tensor(input_ids_list).clone(),
            'attention_mask': torch.ones((len(input_ids_list), max_length)).clone(),
            "position_ids": torch.arange(0, max_length).unsqueeze(0).expand(len(input_ids_list), max_length).clone(),
            'labels': torch.tensor(labels_list).clone(),
        }
        return model_inputs

    
class WYLLamaDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, collate_fn, args_data):
        super().__init__()
        self.datasets = load_dataset(args_data.raw_file_type, data_dir = args_data.data_dir, data_files={'train':'train.json',
                                                                                                        'validation':'valid.json',
                                                                                                        'test':'valid.json'})

        self.args_data = args_data
        self.tokenizer = tokenizer
        self.collate_fn  = collate_fn

    def setup(self, stage: Optional[str] = None) -> None:
        return

    def train_dataloader(self):
        ds = self.datasets['train']

        collate_fn = self.collate_fn
        if hasattr(ds, 'collate_fn'):
            collate_fn = ds.collate_fn

        return DataLoader(
            ds,
            shuffle=True,
            batch_size=self.args_data.train_batchsize,
            num_workers=self.args_data.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        ds = self.datasets['train']
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
        ds = self.datasets['train']
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