"""Modules for creating and running a gpt2 trainer"""

from .models.gpt2 import gpt2_model_gpt2_generator
from .lit_models.base import GPT2_BaseLitModel
from .lit_models.wenzhonglitmodel import WenzhongQALitModel