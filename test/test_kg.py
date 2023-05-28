
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
    kg_model = kg_model_entities_generator(hp.kg.pretrained_file,4)
    #data
    data, tokenizer= setup_data_from_args(hp)

    kg_module = KG_BaseLitModel(kg_model, tokenizer,args=hp.kg)
    trainer = pl.Trainer(accelerator='mps',
                         devices=1,
                        max_epochs=3,
                        )
    
    #trainer.test(combined_module,data)
    logits = kg_module.predict(['蓝方直升机被红方载具击伤撤退，多名步兵由D6绕至D7'])
    print(logits)


if __name__ == "__main__":
    main()