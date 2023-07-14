import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    PeftModel,
)
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2LMHeadModel,GPT2Tokenizer

def LlamaPeftGenerate(args_model,args_lora):
    model = LlamaForCausalLM.from_pretrained(
    #model = GPT2LMHeadModel.from_pretrained(
    args_model.base_model,
    torch_dtype=torch.half,
    cache_dir = args_model.cache_dir,
)
 
    tokenizer = LlamaTokenizer.from_pretrained(args_model.base_model,cache_dir = args_model.cache_dir)
    #tokenizer = GPT2Tokenizer.from_pretrained(args_model.base_model,cache_dir = args_model.cache_dir)
    return loraTheModel(model,args_model.base_model,args_lora), tokenizer
    #return model, tokenizer

def loraTheModel(model,model_id,args_lora):
    #model = prepare_model_for_int8_training(model)
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