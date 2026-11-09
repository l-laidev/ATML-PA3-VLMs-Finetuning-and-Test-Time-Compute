import argparse
import os
import json
from pathlib import Path
import torch

from task1_finetuning.utils import set_seed
from task1_finetuning.diagnostics import compute_weight_space_diagnostics, compute_model_displacement
from task1_finetuning.model import load_qwen_model
from task1_finetuning.evaluation import evaluate_model_on_gsm8k


def parse_args():
    parser = argparse.ArgumentParser(description="PA3 Model Evaluation and Diagnostics Entry Point")
    
    parser.add_argument("--variant", type=str, required=True, choices=["base_lm", "full_ft", "full_ft_mid", "l2_sp", "lora_r4", "lora_r16", "oplora_r16"])
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint directory."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Path or identifier of the starting base model for weight-space comparisons."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/task1",
        help="Path to save evaluation metrics and diagnostic JSON output."
    )
    parser.add_argument(
        "--run_weight_diagnostics",
        action="store_true",
        help="Whether to compute weight-space diagnostics (eta, rho_k) on down_proj layers."
    )
    
    return parser.parse_args()


def main():
    set_seed(6304)  # Enforce global determinism
    args = parse_args()
    base_dir = Path(args.output_dir) / args.variant
    
    print("=" * 60)
    print(f"Starting PA3 Evaluation & Diagnostics")
    print(f"  Checkpoint   : {args.checkpoint_path}")
    print(f"  Base Model   : {args.base_model_name}")
    print(f"  Output Directory  : {args.output_dir}")
    print("=" * 60)

    results = {"checkpoint_path": args.checkpoint_path, "metrics": {}}

    if args.run_weight_diagnostics:
        print("Loading base model and fine-tuned checkpoint for weight-space diagnostics...")
        base_model, _ = load_qwen_model(args.base_model_name, dtype=torch.float32)
        finetuned_model, _ = load_qwen_model(args.checkpoint_path, dtype=torch.float32)

        layer_diagnostics = {}
        
        # Target layers specified in assignment: down_proj in layers 0, 13, 27
        target_layers = [0, 13, 27]
        
        with torch.no_grad():
            for layer_idx in target_layers:
                base_layer = base_model.model.layers[layer_idx].mlp.down_proj
                ft_layer = finetuned_model.model.layers[layer_idx].mlp.down_proj
                
                W_0 = base_layer.weight.data.float()
                W_t = ft_layer.weight.data.float()
                
                U, S, Vh = torch.linalg.svd(W_0, full_matrices=False)
                # U_k (k=128)
                k = min(128, U.size(1))
                U_k = U[:, :k]
                
                diag_metrics = compute_weight_space_diagnostics(W_0, W_t, U_k)
                layer_diagnostics[f"layer_{layer_idx}"] = diag_metrics
                print(f"  Layer {layer_idx} down_proj -> eta: {diag_metrics['eta']:.5f}, rho_k: {diag_metrics['rho_k']:.5f}")

        results["metrics"]["layer_diagnostics"] = layer_diagnostics

        results["metrics"]["full_displacement"] = compute_model_displacement(base_model.named_parameters(), finetuned_model)

    os.makedirs(base_dir, exist_ok=True)
    with open(base_dir / "weight_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation diagnostics successfully saved")

    evaluate_model_on_gsm8k(args.checkpoint_path, base_dir / "benchmarks")


if __name__ == "__main__":
    main()