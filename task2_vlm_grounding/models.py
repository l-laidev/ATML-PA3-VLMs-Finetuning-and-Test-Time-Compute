import torch
import torch.nn as nn

from transformers import AutoProcessor, SiglipVisionModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model



def load_connector():
    return nn.Sequential(
        nn.Linear(768, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
    ).to("cuda" if torch.cuda.is_available() else "cpu").to(torch.float32)

def load_vision_encoder():
    model_id = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = SiglipVisionModel.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    for p in model.parameters():
        p.requires_grad_(False)
    
    model.eval()
    return processor, model

def load_llm():
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    for p in model.parameters():
        p.requires_grad_(False)
    
    return tokenizer, model

def make_lora(llm):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm = get_peft_model(llm, peft_config)
    llm.print_trainable_parameters()
    
    return llm
