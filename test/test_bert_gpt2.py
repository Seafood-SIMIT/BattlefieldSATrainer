
import sys
sys.path.append('.')
from utils import HParam
from kg_generator import *
from gpt2_generator import gpt2_model_gpt2_generator, GPT2_BaseLitModel
from combined_generator.lit_models.base import CombinedBaseModule
from training.util import import_class, setup_data_from_args

import argparse
import pytorch_lightning as pl


def main():
    parser = argparse.ArgumentParser(add_help=False)


    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("-c",'--config',default='config/default.yaml', type=str, help='set the config file')
    parser.add_argument("-m", type=str, required= True, help='model name')

    args = parser.parse_args()
    hp = HParam(args.config)

    #kg_model = kg_model_transformer_generator(hp.kg.pretrained_file)
    kg_model = kg_model_entities_generator(hp.kg.pretrained_file,hp.kg.num_labels)
    gpt2_model = gpt2_model_gpt2_generator(hp.gpt2.pretrained_file)
    #data
    data, tokenizer= setup_data_from_args(hp)

    combined_module = CombinedBaseModule(kg_model,gpt2_model,tokenizer)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                        max_epochs=3,
                        )
    
    #trainer.test(combined_module,data)
    combined_module.test_step(['蓝方直升机被红方载具击伤撤退，多名步兵由D6绕至D7'])


if __name__ == "__main__":
    main()