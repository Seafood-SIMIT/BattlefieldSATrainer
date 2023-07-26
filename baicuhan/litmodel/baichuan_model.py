from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from peft import (
    LoraConfig,
    get_peft_model,
)
from typing import List, Optional

from transformers.modeling_utils import PreTrainedModel
# Includes: (1) cast the layernorm in fp32 (2) make output embedding layer require grads (3) upcast the lm_head to fp32
# Inspired by: https://github.com/huggingface/peft/blob/c0209c35abbf88c63aa267800d98a8e212ed0a42/src/peft/utils/other.py#L35


def loadModelTokenizer(args,args_lora):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=args.use_fast,
        cache_dir = args.cache_dir
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token

    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.compute_dtype == 'torch.bfloat16' else torch.float16,
        cache_dir = args.cache_dir
    )

    model = loraTheModel(model,args_lora)
    return model, tokenizer


def loraTheModel(model,args_lora):
    #model = prepare_model_for_training(model)
    config = LoraConfig(
        r=args_lora.lora_r,
        lora_alpha=args_lora.lora_alpha,
        target_modules=args_lora.lora_target_modules,
        lora_dropout=args_lora.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    #model.enable_input_require_grads()
    #model = PeftModel.from_pretrained(model,model_id, is_trainable=True)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

