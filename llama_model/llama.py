import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel,GPT2Tokenizer

def LlamaPeftGenerate(args_model,args_lora):
    model =AutoModelForCausalLM.from_pretrained(
    #model = GPT2LMHeadModel.from_pretrained(
    args_model.base_model,
    trust_remote_code=True,
    #torch_dtype=torch.bfloat16,
    cache_dir = args_model.cache_dir,
)
 
    tokenizer =AutoTokenizer.from_pretrained(args_model.base_model,trust_remote_code=True,cache_dir = args_model.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    #tokenizer = GPT2Tokenizer.from_pretrained(args_model.base_model,cache_dir = args_model.cache_dir)
    #whether to use lora
    if args_model.do_evalonly:
        return model, tokenizer
        
    else:
        return loraTheModel(model,args_model.base_model,args_lora), tokenizer

def loraTheModel(model,model_id,args_lora):
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