import argparse

from transformers import GPT2LMHeadModel
class GPT2(GPT2LMHeadModel):
    def __init__(self, model_path):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

def gpt2_model_gpt2_generator(final_path):
    print('Loading GPT2 Model')
    return GPT2LMHeadModel.from_pretrained(final_path)