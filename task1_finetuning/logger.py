import os
import json
import time
import torch
from typing import Dict, Any

class TrainingLogger:
    def __init__(self, output_dir: str, variant: str):
        self.output_dir = output_dir
        self.variant = variant
        self.start_time = time.time()
        self.metrics_history = []
        os.makedirs(output_dir, exist_ok=True)

    def log_skipped_update(self, step: int):
        self.log_step(step, None, None, None, None, None, is_skipped=True)

    def log_step(self, step: int, sft_loss: float, lr: float, l2sp_penalty: float = 0.0, lambda_l2sp_penalty: float = 0.0, total_loss: float = 0.0, is_skipped=False):
        record = {
            "step": step,
            "sft_loss": sft_loss,
            "learning_rate": lr,
            "l2sp_omega": l2sp_penalty,
            "lambda_l2sp_omega": lambda_l2sp_penalty,
            "total_loss": total_loss if total_loss != 0.0 else sft_loss,
            "is_skipped": is_skipped,
        }
        self.metrics_history.append(record)

    def finalize_run(self, model, ckpt_dir: str, initial_params: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        end_time = time.time()
        training_time_sec = end_time - self.start_time
        
        # Peak GPU memory
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        # Trainable parameter count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Checkpoint size on disk
        ckpt_size_mb = 0.0
        for root, _, files in os.walk(ckpt_dir):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    ckpt_size_mb += os.path.getsize(fp) / (1024 * 1024)

        # Full-model parameter displacement ||theta - theta_0||_2 if initial_params provided
        model_displacement = 0.0
        if initial_params is not None:
            sq_diff = 0.0
            for name, param in model.named_parameters():
                if name in initial_params:
                    theta_0 = initial_params[name].to(param.device).float()
                    sq_diff += torch.sum((param.float() - theta_0) ** 2).item()
            model_displacement = float(torch.sqrt(torch.tensor(sq_diff)).item())

        summary = {
            "variant": self.variant,
            "training_time_minutes": training_time_sec / 60.0,
            "peak_gpu_memory_mb": peak_gpu_mem_mb,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "checkpoint_size_mb": ckpt_size_mb,
            "model_displacement_l2": model_displacement
        }

        # Save metrics history and summary
        log_data = {
            "summary": summary,
            "history": self.metrics_history
        }
        
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
            
        print(f"\n[Training Summary: {self.variant}]")
        for k, v in summary.items():
            print(f"  {k}: {v}")
            
        return summary