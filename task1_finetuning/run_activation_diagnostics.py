import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from task1_finetuning.utils import set_seed
from task1_finetuning.probes import get_or_create_probe_set
from task1_finetuning.diagnostics import compute_relative_activation_change


def main():
    set_seed(6304)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    probe_payload = get_or_create_probe_set(tokenizer)
    all_probes = probe_payload["math_probes"] + probe_payload["non_math_probes"]
    
    print(f"Loaded {len(all_probes)} probes for activation diagnostics.")

    print("Loading base model to cache input activations z...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.float32, device_map="auto")
    base_model.eval()

    target_layers = [0, 13, 27]
    cached_base_activations = {layer_idx: [] for layer_idx in target_layers}
    
    # Cache hooks
    hooks = []
    def make_hook(layer_idx):
        def hook(module, input, output):
            z = input[0].detach().float()[:, -1, :].cpu() # Last non-padding token
            cached_base_activations[layer_idx].append(z)
        return hook

    for idx, block in enumerate(base_model.model.layers):
        if idx in target_layers:
            h = block.mlp.down_proj.register_forward_hook(make_hook(idx))
            hooks.append(h)

    print("Running base model inference on probes to cache z...")
    with torch.no_grad():
        for probe in all_probes:
            inputs = tokenizer(probe["prompt"], return_tensors="pt").to(device)
            base_model(**inputs)

    for h in hooks:
        h.remove()
    del base_model
    torch.cuda.empty_cache()

    # Checkpoints to evaluate R(z)
    checkpoints = {
        "full_ft_final": "./checkpoints/task1/full_ft/checkpoint_update_1250_examples_40000",
        "full_ft_mid": "./checkpoints/task1/full_ft/checkpoint_update_625_examples_20000",
        "l2_sp_final": "./checkpoints/task1/l2_sp/checkpoint_update_1250_examples_40000",
        "lora_r4_final": "./checkpoints/task1/lora_r4/merged_model",
        "lora_r16_final": "./checkpoints/task1/lora_r16/merged_model",
        "oplora_r16_final": "./checkpoints/task1/oplora_r16/merged_model"
    }

    # Load base weights W_0 for comparison
    temp_base, _ = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.float32), None
    base_weights = {layer_idx: temp_base.model.layers[layer_idx].mlp.down_proj.weight.data.float() for layer_idx in target_layers}
    del temp_base

    results = {}

    for cond_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        print(f"Computing activation diagnostics for {cond_name}...")
        ft_model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=torch.float32, device_map="auto")
        ft_model.eval()

        cond_results = {layer_idx: {"math": [], "non_math": []} for layer_idx in target_layers}

        with torch.no_grad():
            for layer_idx in target_layers:
                W_0 = base_weights[layer_idx]
                W_t = ft_model.model.layers[layer_idx].mlp.down_proj.weight.data.float()
                
                for p_idx, probe in enumerate(all_probes):
                    z = cached_base_activations[layer_idx][p_idx]
                    r_val = compute_relative_activation_change(z.to(W_t.device), W_0.to(W_t.device), W_t)
                    
                    if "gsm8k" in probe["id"]:
                        cond_results[layer_idx]["math"].append(r_val)
                    else:
                        cond_results[layer_idx]["non_math"].append(r_val)

        # Average across probes
        summary_r = {}
        for layer_idx in target_layers:
            m_avg = sum(cond_results[layer_idx]["math"]) / len(cond_results[layer_idx]["math"]) if cond_results[layer_idx]["math"] else 0
            nm_avg = sum(cond_results[layer_idx]["non_math"]) / len(cond_results[layer_idx]["non_math"]) if cond_results[layer_idx]["non_math"] else 0
            summary_r[f"layer_{layer_idx}"] = {"math_mean_R": m_avg, "non_math_mean_R": nm_avg}

        results[cond_name] = summary_r
        del ft_model
        torch.cuda.empty_cache()

    base_dir = "results/task1/"
    os.makedirs(base_dir, exist_ok=True)

    output_file = base_dir + "probe_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Activation diagnostics successfully saved to {output_file}")


if __name__ == "__main__":
    main()