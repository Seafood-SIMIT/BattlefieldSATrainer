
import sys
sys.path.append('.')
from utils import HParam
from kg_generator import *
from gpt2_generator import gpt2_model_gpt2_generator, GPT2_BaseLitModel
from training.util import import_class, setup_data_from_args

import argparse
import torch
import pytorch_lightning as pl


def main():
    parser = argparse.ArgumentParser(add_help=False)


    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("-c",'--config',default='config/default.yaml', type=str, help='set the config file')
    parser.add_argument("-m", type=str, required= True, help='model name')

    args = parser.parse_args()
    hp = HParam(args.config)

    #kg_model = kg_model_transformer_generator(hp.kg.pretrained_file)
    gpt2_model = gpt2_model_gpt2_generator(hp.gpt2.pretrained_file)
    #data
    data, tokenizer= setup_data_from_args(hp)

    gpt2_litmodel = GPT2_BaseLitModel(gpt2_model, args=hp.gpt2)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                        max_epochs=3,
                        )
    
    #trainer.test(combined_module,data)
    generation_outputs = gpt2_litmodel.predict(tokenizer('蓝方直升机被红方载具击伤撤退，多名步兵由D6绕至D7',return_tensors='pt'))

    for idx, sentense in enumerate(generation_outputs.sequences):
        print('next sentence %d:\n'%idx,
        tokenizer.decode(sentense).split('<|endoftext|>')[0])
        print('*'*40)



if __name__ == "__main__":
    main()