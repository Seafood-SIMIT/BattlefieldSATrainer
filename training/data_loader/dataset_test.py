import torch
from torch.utils.data import Dataset
class KGDataset(Dataset):
    def __init__(self, data):
        self.data=data
        
        ##划分测试集
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text=self.data[idx]["text"]
        input_ids = self.data[idx]['input_ids'].squeeze(0)
        attention_mask = self.data[idx]['attention_mask'].squeeze(0)
        label_ids = self.data[idx]['label']
        return input_ids, attention_mask, label_ids
