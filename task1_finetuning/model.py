import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def load_qwen_model(model_name: str = "Qwen/Qwen3-0.6B", dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def configure_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


class OPLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 32, k: int = 128, device: torch.device | str = "cuda"):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.k = k
        
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        d_out, d_in = base_layer.weight.shape
        
        print(f"Computing SVD for weight matrix of shape ({d_out}, {d_in}) with protection rank k={k}...")
        with torch.no_grad():
            W_float = base_layer.weight.data.float()
            U, S, Vh = torch.linalg.svd(W_float, full_matrices=False)
            
            actual_k = min(k, U.size(1), Vh.size(0))
            U_k = U[:, :actual_k]
            V_k = Vh[:actual_k, :].T
            
        self.register_buffer("U_k", U_k)
        self.register_buffer("V_k", V_k)
        
        self.lora_A = nn.Parameter(torch.empty(r, d_in)).to(device)
        self.lora_B = nn.Parameter(torch.empty(d_out, r)).to(device)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # same as in LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        
        b_proj = self.U_k @ (self.U_k.T @ self.lora_B)
        # B - U U^T B = (I - U U^T) B = PB
        tilde_B = self.lora_B - b_proj
        
        a_proj = (self.lora_A @ self.V_k) @ self.V_k.T
        # A - A V V^T = A (I - V V^T) = AP
        tilde_A = self.lora_A - a_proj
        
        scaling = self.alpha / self.r
        lora_output = scaling * (nn.functional.linear(x, tilde_A) @ tilde_B.T)
        
        return base_output + lora_output.to(base_output.dtype)


def convert_model_to_oplora(model, r: int = 16, alpha: int = 32, k: int = 128):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for name, module in list(model.named_modules()):
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            
            oplora_layer = OPLoRALinear(module, r=r, alpha=alpha, k=k, device=model.device)
            setattr(parent, child_name, oplora_layer)
            
    return model