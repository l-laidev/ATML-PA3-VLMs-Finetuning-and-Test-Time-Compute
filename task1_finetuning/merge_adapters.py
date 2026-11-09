import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file
import os

from task1_finetuning.model import OPLoRALinear, convert_model_to_oplora, load_qwen_model
from task1_finetuning.utils import set_seed


def merge_adapters_into_fresh_backbone(fresh_model, checkpoint_path, variant):
    if "oplora" in variant:
        print(f"Merging OPLoRA projected factors from {checkpoint_path}...")
        merged_model = convert_model_to_oplora(fresh_model, r=16, alpha=32, k=128)
        
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
            
        merged_model.load_state_dict(state_dict, strict=False)
        
        with torch.no_grad():
            for name, module in merged_model.named_modules():
                if isinstance(module, OPLoRALinear):
                    # projected factors
                    tilde_B = module.lora_B - module.U_k @ (module.U_k.T @ module.lora_B)
                    tilde_A = module.lora_A - (module.lora_A @ module.V_k) @ module.V_k.T
                    
                    delta_w = (module.alpha / module.r) * (tilde_B @ tilde_A)
                    
                    # add delta_w, cast to base dtype
                    module.base_layer.weight.data += delta_w.to(module.base_layer.weight.dtype)
        
        # unwrap back to standard nn.Linear for clean inference
        merged_model = unwrap_oplora_layers(merged_model)
    elif "lora" in variant:
        print(f"Merging PEFT LoRA from {checkpoint_path}...")
        model_to_merge = PeftModel.from_pretrained(fresh_model, checkpoint_path)
        merged_model = model_to_merge.merge_and_unload()    
    
    merged_model.eval()
    return merged_model

def unwrap_oplora_layers(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, OPLoRALinear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.base_layer)
    return model

@torch.no_grad()
def verify_merged_logits(unmerged_model, merged_model, tokenizer, device="cuda"):
    unmerged_model.eval()
    merged_model.eval()
    
    # dummy prompt batch
    test_texts = [
        "Solve for x: 2x + 5 = 15",
        "What is the derivative of x^2?"
    ]
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # unmerged model
    outputs_unmerged = unmerged_model(**inputs)
    logits_unmerged = outputs_unmerged.logits
    
    # merged model
    outputs_merged = merged_model(**inputs)
    logits_merged = outputs_merged.logits
    
    max_diff = torch.max(torch.abs(logits_unmerged - logits_merged)).item()
    mean_diff = torch.mean(torch.abs(logits_unmerged - logits_merged)).item()
    
    print(f"--- Logit Verification Check ---")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    
    # FP16 precision threshold
    if max_diff < 1e-3:
        print(" SUCCESS: Merged and unmerged logits match within expected FP16 tolerance!")
    else:
        print(" WARNING: Logit difference is higher than expected. Check scaling factors or projection logic.")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, required=True, choices=["lora_r4", "lora_r16", "oplora_r16"])
    parser.add_argument("--output_root", type=str, default="./checkpoints")
    
    return parser.parse_args()

def main():
    set_seed(6304)
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = os.path.join(args.output_root, "task1", args.variant)
    fresh_model, tokenizer = load_qwen_model(args.model_name, dtype=torch.float16)
    
    ckp_path = Path(output_dir) / args.checkpoint
    merged_path = Path(output_dir) / "merged_model"

    merged_model = merge_adapters_into_fresh_backbone(fresh_model, ckp_path, args.variant)
    
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    verify_merged_logits(fresh_model, merged_model, tokenizer, device)


if __name__ == "__main__":
    main()