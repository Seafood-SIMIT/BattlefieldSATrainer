import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import LlamaTokenizer, LlamaForCausalLM

def llamaModelGenerate(args_llama,args_lora):
    model = LlamaForCausalLM.from_pretrained(
    args_llama.base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    cache_dir = args_llama.cache_dir,
)
 
    tokenizer = LlamaTokenizer.from_pretrained(args_llama.base_model)
    return loraTheModel(model,args_lora), tokenizer

def loraTheModel(model,args_lora):
    model = prepare_model_for_int8_training(model)
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