import os
from pathlib import Path
import random
import numpy as np
from peft import PeftModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, DataCollatorForSeq2Seq
from safetensors.torch import load_file
from typing import Dict, Optional
import tqdm

from task1_finetuning.logger import TrainingLogger
from task1_finetuning.model import OPLoRALinear


class PA3Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        output_dir: str = "./checkpoints",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        lambda_l2sp: float = 0.0,
        initial_params: Optional[Dict[str, torch.Tensor]] = None,
        device: str = "cuda",
        logger = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_l2sp = lambda_l2sp
        self.initial_params = initial_params
        self.device = device
        self.logger = logger
        
        os.makedirs(output_dir, exist_ok=True)

    def save_checkpoint(self, ckpt_path, global_step, epoch, step, optimizer, scheduler, scaler, logger: TrainingLogger):
        os.makedirs(ckpt_path, exist_ok=True)
                                
        # model weights and tokenizer config
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        
        # RNG states
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }

        # logger state
        logger_state = {
            "start_time": logger.start_time,
            "history": logger.metrics_history
        }
        
        # training state dictionary
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_states": rng_states,
            "logger_state": logger_state,
        }
        
        state_file_path = os.path.join(ckpt_path, "training_state.pt")
        torch.save(training_state, state_file_path)

    def resume_from_checkpoint(self, ckpt_path: str, logger: TrainingLogger):
        print(f"Resuming training from checkpoint: {ckpt_path}")
        
        state_file_path = os.path.join(ckpt_path, "training_state.pt")
        if not os.path.exists(state_file_path):
            raise FileNotFoundError(f"Training state file not found at {state_file_path}")
        
        training_state = torch.load(state_file_path, map_location=self.device, weights_only=False)
        
        # RNG states
        rng_states = training_state["rng_states"]
        random.setstate(rng_states["python"])
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in rng_states["torch_cuda"]])

        # logger state
        logger_state = training_state["logger_state"]
        logger.start_time = logger_state["start_time"]
        logger.metrics_history = logger_state["history"]
        
        start_global_step = training_state["global_step"]
        start_epoch = training_state["epoch"]
        last_step_in_epoch = training_state["step"]
        
        print(f"Restored state -> Epoch: {start_epoch}, Global Step: {start_global_step}, Step in Epoch: {last_step_in_epoch}")
        
        return start_epoch, start_global_step, last_step_in_epoch

    def compute_l2sp_penalty(self) -> torch.Tensor:
        """
        Sum over all unique parameters including biases and normalization parameters, counting tied parameters once.
        """
        if self.initial_params is None or self.lambda_l2sp == 0.0:
            return torch.tensor(0.0, device=self.device)
            
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.initial_params:
                theta_0 = self.initial_params[name].to(param.device).to(param.dtype)
                penalty += torch.sum((param - theta_0) ** 2)
                del theta_0
        return 0.5 * penalty

    def train(self, num_epochs: int = 1, batch_size: int = 2, gradient_accumulation_steps: int = 16, max_updates: int = 1250, resume_from_checkpoint_path: str | None = None):
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8,  # for Tensor Core optimization on GPUs
            label_pad_token_id=-100,
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,  # do not shuffle retained examples
            collate_fn=data_collator,
            generator=torch.Generator().manual_seed(6304),
            pin_memory=True,
        )
        
        decay_parameters = []
        no_decay_parameters = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name:
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)
                
        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=38,
            num_training_steps=max_updates
        )
        
        scaler = torch.amp.GradScaler("cuda")
        self.model.train()
        optimizer.zero_grad()
        
        start_epoch = 0
        global_step = 0
        skip_steps = -1
        accumulated_loss = 0.0

        if resume_from_checkpoint_path is not None and Path(resume_from_checkpoint_path).exists():
            print(f"Resuming training from checkpoint: {resume_from_checkpoint_path}")
            if isinstance(self.model, PeftModel):
                # reload adapter weights
                self.model = PeftModel.from_pretrained(
                    self.model.base_model.model, 
                    resume_from_checkpoint_path, 
                    is_trainable=True
                )
            elif any(isinstance(m, OPLoRALinear) for m in self.model.modules()):
                weights_file = os.path.join(resume_from_checkpoint_path, "model.safetensors")
                if os.path.exists(weights_file):
                    state_dict = load_file(weights_file)
                else:
                    weights_file = os.path.join(resume_from_checkpoint_path, "pytorch_model.bin")
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
                
                self.model.load_state_dict(state_dict, strict=False)
            else:
                # for Full FT / L2-SP, reload the state dict or model weights
                model_class = type(self.model)
                self.model = model_class.from_pretrained(resume_from_checkpoint_path, dtype=torch.float32)
            
            self.model.to(self.device)

            start_epoch, global_step, skip_steps = self.resume_from_checkpoint(resume_from_checkpoint_path, self.logger)
            
            # optimizer, scheduler, and scaler state
            state_file_path = os.path.join(resume_from_checkpoint_path, "training_state.pt")
            training_state = torch.load(state_file_path, map_location=self.device, weights_only=False)
            
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            scaler.load_state_dict(training_state["scaler_state_dict"])
        
        checkpoint_update_milestones = {1: "32", 312: "9984", 625: "20000", 937: "29984", 1250: "40000"}
        
        print(f"Starting training for {max_updates} updates (effective batch size 32)...")
        
        for epoch in range(num_epochs):
            if epoch < start_epoch:
                continue  # skip completed epochs

            for step, batch in (pbar := tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training step")):
                if epoch == start_epoch and step <= skip_steps:
                    continue  # already processed in the start_epoch

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    sft_loss = outputs.loss
                    
                    total_loss = sft_loss
                    l2sp_pen = None
                    if self.lambda_l2sp > 0.0:
                        l2sp_pen = self.compute_l2sp_penalty()
                        total_loss = sft_loss + self.lambda_l2sp * l2sp_pen
                        
                    loss = total_loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                accumulated_loss += sft_loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    scale_before = scaler.get_scale()
                    scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()

                    optimizer_skipped = scale_after < scale_before
                    if optimizer_skipped:
                        pbar.clear()
                        print(f"Update {global_step + 1}: Gradient overflow detected (Inf/NaN). Update skipped. Scale: {scale_before} -> {scale_after}")
                        pbar.refresh()
                        if self.logger is not None:
                            self.logger.log_skipped_update(step=global_step + 1)
                    
                    # Step scheduler only if update wasn't skipped (avoids PyTorch warnings)
                    if not optimizer_skipped:
                        scheduler.step()

                    optimizer.zero_grad()
                    global_step += 1
                    current_lr = scheduler.get_last_lr()[0]

                    if self.logger is not None:
                        self.logger.log_step(
                            step=global_step,
                            sft_loss=sft_loss.item(),
                            lr=current_lr,
                            l2sp_penalty=l2sp_pen.item() if isinstance(l2sp_pen, torch.Tensor) else l2sp_pen,
                            lambda_l2sp_penalty=self.lambda_l2sp * (l2sp_pen.item() if isinstance(l2sp_pen, torch.Tensor) else l2sp_pen) if self.lambda_l2sp != 0.0 else 0.0,
                            total_loss=total_loss.item()
                        )
                    
                    if global_step % 10 == 0:
                        pbar.clear()
                        print(f"Update {global_step}/{max_updates} | SFT Loss: {accumulated_loss / 10:.4f}")
                        accumulated_loss = 0.0
                        pbar.refresh()
                        
                    if global_step in checkpoint_update_milestones:
                        ckpt_name = f"checkpoint_update_{global_step}_examples_{checkpoint_update_milestones[global_step]}"
                        ckpt_path = os.path.join(self.output_dir, ckpt_name)
                        
                        pbar.clear()
                        print(f"Saving recovery checkpoint at update {global_step} to {ckpt_path}")
                        pbar.refresh()
                        
                        self.save_checkpoint(ckpt_path, global_step, epoch, step, optimizer, scheduler, scaler, self.logger)
                        
                    if global_step >= max_updates:
                        break
            if global_step >= max_updates:
                break
        
        if self.logger is not None:
            final_ckpt_dir = os.path.join(self.output_dir, f"checkpoint_update_{global_step}_examples_{checkpoint_update_milestones[global_step]}")
            self.logger.finalize_run(self.model, final_ckpt_dir, self.initial_params)
        
        print("Training completed successfully.")