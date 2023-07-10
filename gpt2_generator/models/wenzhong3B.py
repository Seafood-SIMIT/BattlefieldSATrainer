import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import LlamaTokenizer, LlamaForCausalLM

def wenzhong3BPeftGenerate(args_model,args_lora):
    model = LlamaForCausalLM.from_pretrained(
    args_model.base_model,
    torch_dtype=torch.half,
    cache_dir = args_model.cache_dir,
)
 
    tokenizer = LlamaTokenizer.from_pretrained(args_model.base_model,cache_dir = args_model.cache_dir)
    return loraTheModel(model,args_lora), tokenizer

def loraTheModel(model,args_lora):
    #model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args_lora.lora_r,
        lora_alpha=args_lora.lora_alpha,
        target_modules=args_lora.lora_target_modules,
        lora_dropout=args_lora.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model