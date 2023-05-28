"""Utilities for model development scripts: training and staging."""
import argparse
import importlib
#from .data_loader.tokenizer import BertTokenizer
from transformers import GPT2Tokenizer




def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_from_args(hp: argparse.Namespace):
    data_class = import_class(f"{hp.data.data_class_module}.{hp.data.data_class}")

    #tokenizer = BertTokenizer(vocab_file = hp.tokenizer.tokenizer_path)

    tokenizer = GPT2Tokenizer.from_pretrained(hp.tokenizer.tokenizer_path)


    data = data_class(hp.data, tokenizer)
    #model = model_class(model_path=args.litmodel.final_path)

    return data, tokenizer
