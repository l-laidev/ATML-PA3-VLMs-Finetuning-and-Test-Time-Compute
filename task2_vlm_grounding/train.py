import os
from pathlib import Path
import time

import torch
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from task2_vlm_grounding.utils import append_jsonl, load_ckpt, normalize_answer, prepare_batch, process_batch_loss, save_ckpt
from task2_vlm_grounding.data.multitask_sampler import MultiTaskCurriculumSampler
from task2_vlm_grounding.evaluation import evaluate_dev


@torch.no_grad()
def calibrate_connector_magnitude(
    connector, llm, tokenizer, train_loader,
    num_batches=10,
    log_path="results/task2/calibration.json"
):
    connector.eval()
    llm.eval()
    
    print("Running connector magnitude calibration on training data...")
    text_norms = []
    visual_norms = []
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        if batch_idx >= num_batches:
            break

        images, captions = batch

        prompts = ["Describe this image." for _ in range(len(captions))]
        visual_embeds = connector(images.to(llm.device, dtype=torch.float32)).to(torch.float16)
        text_embeds, attn_mask, _ = prepare_batch(llm, tokenizer, prompts, captions, None, is_caption=True)

        valid_text_embeds = text_embeds[attn_mask == 1].view(-1, text_embeds.shape[-1])
        text_norms.append(torch.norm(valid_text_embeds, p=2, dim=-1).mean().item())
        visual_norms.append(torch.norm(visual_embeds, p=2, dim=-1).mean().item())

    # empirical expectations
    mu_text = sum(text_norms) / len(text_norms)
    mu_visual = sum(visual_norms) / len(visual_norms)
    
    # scaling factor rule
    scaling_factor = mu_text / mu_visual
    
    print(f"[Calibration Results]")
    print(f"  - Mean Text Embedding L2 Norm (mu_text): {mu_text:.4f}")
    print(f"  - Mean Visual Embedding L2 Norm (mu_visual): {mu_visual:.4f}")
    print(f"  - Derived Scaling Factor (s): {scaling_factor:.4f}")

    # apply scaling factor directly to the final weights
    with torch.no_grad():
        connector[-1].weight.mul_(scaling_factor)
        connector[-1].bias.mul_(scaling_factor)
    
    # Record rule and resulting scale
    calibration_record = {
        "rule": "s = mu_text / mu_visual (L2 norm expectation matching)",
        "mu_text": mu_text,
        "mu_visual_uncalibrated": mu_visual,
        "scaling_factor": scaling_factor,
        "mu_visual_calibrated": mu_visual * scaling_factor,
        "samples_used": num_batches * images.shape[0]
    }

    append_jsonl(log_path, calibration_record)
    
    return scaling_factor

def train_model(
    connector, llm, tokenizer, train_loader, dev_loader, stage: int, 
    total_steps: int, connector_lr: float, llm_lr: float = None, 
    accum_steps: int = 2, batch_size: int = 8, ckpt_dir: str = "./checkpoints/task2/", dev_dir: str = "results/task2/dev",
    resume_training: bool = True, prev_stage_updates: int = 0,
):
    ckpt_path = Path(f"{ckpt_dir}/stage-{stage}/")
    ckpt_path.mkdir(parents=True, exist_ok=True)

    dev_log_path = Path(dev_dir) / f"dev.jsonl"
    dev_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    param_groups = [{"params": connector.parameters(), "lr": connector_lr}]
    llm_trainable_params = []
    
    if llm_lr is not None:
        llm_trainable_params = [p for p in llm.parameters() if p.requires_grad]
        param_groups.append({"params": llm_trainable_params, "lr": llm_lr})
    
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )
    
    # decay to 10% of peak LR
    decay_to = 0.10
    for i, lmbda in enumerate(scheduler.lr_lambdas):
        scheduler.lr_lambdas[i] = lambda step, l=lmbda: decay_to + (1.0 - decay_to) * l(step)
    
    connector.train()
    if stage == 1:
        llm.eval()
    else:
        llm.train()
    opt.zero_grad(set_to_none=True)

    connector_total = sum(p.numel() for p in connector.parameters())
    llm_total = sum(p.numel() for p in llm.parameters())
    connector_trainable = sum(p.numel() for p in connector.parameters() if p.requires_grad)
    llm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    
    global_step = 0
    step = 0
    resume_step = -1
    logs = []
    time_ref = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    
    if resume_training and ckpt_path.exists() and len(os.listdir(ckpt_path)) != 0:
        resume_step, global_step, logs = load_ckpt(ckpt_path, connector, opt, scheduler, scaler, llm, stage)
        print(f"Resumed training from step {resume_step}, global_step {global_step}")
    
    progress_bar = tqdm(total=total_steps, initial=global_step, desc=f"Stage {stage} Training")
    is_sampler = isinstance(train_loader, MultiTaskCurriculumSampler)
    iterator = iter(train_loader) if not is_sampler else None

    if stage == 1:
        evaluate_dev(connector, llm, tokenizer, dev_loader, 0, stage, dev_log_path)
    
    assert batch_size % 2 == 0
    total_loss_per_opt = 0.0
    while global_step < total_steps:
        if is_sampler:
            batch_items = [train_loader.sample() for _ in range(batch_size // 2)]  # every entry is a pair, so they give 2x samples
            images, prompts, targets, tasks = [], [], [], []
            for task, item in batch_items:
                for pair in item:
                    if task != "vqa":
                        images.append(pair[0])
                        prompts.append("" if task == "caption" else pair[1])
                        targets.append(pair[1] if task == "caption" else pair[2])
                        tasks.append(task)
                    else:
                        images.append(pair["image"])
                        prompts.append(pair["question"])
                        targets.append(pair["target"])
                        tasks.append(task)
            images = torch.stack(images, dim=0)
        else:
            try:
                images_batch, captions_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images_batch, captions_batch = next(iterator)
            
            images = images_batch # [B, 196, 768]
            targets = list(captions_batch)
            prompts = ["Describe this image." for _ in range(len(targets))]
            tasks = ["caption"] * len(prompts)
        
        step += 1
        if step <= resume_step:
            continue

        # assert images.shape[0] == batch_size, f"{images.shape[0]} vs {batch_size}"
        # assert len(targets) == batch_size, f"{len(targets)} vs {batch_size}"
        # assert len(prompts) == batch_size, f"{len(prompts)} vs {batch_size}"

        with torch.amp.autocast("cuda", dtype=torch.float16):
            # "images" are actually extracted features
            visual_embeds = connector(images.to(llm.device, dtype=torch.float32)).to(torch.float16)
            batch_loss = process_batch_loss(llm, tokenizer, prompts, targets, visual_embeds) / accum_steps

        total_loss_per_opt += batch_loss.detach().cpu().item()
        scaler.scale(batch_loss).backward()
        
        if step % accum_steps == 0:
            scale_before = scaler.get_scale()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(connector.parameters(), max_norm=1.0)
            if llm_lr is not None and len(llm_trainable_params) > 0:
                torch.nn.utils.clip_grad_norm_(llm_trainable_params, max_norm=1.0)
            
            scaler.step(opt)
            scaler.update()

            scale_after = scaler.get_scale()
            optimizer_skipped = scale_after < scale_before

            if not optimizer_skipped:
                scheduler.step()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            
            # based on main-run updates, not stage-specific updates
            main_run_updates = global_step + prev_stage_updates
            
            if main_run_updates % 500 == 0 or global_step == total_steps:
                evaluate_dev(connector, llm, tokenizer, dev_loader, main_run_updates, stage, dev_log_path)

            logs.append({
                "loss": total_loss_per_opt,
                "stage": stage,
                "main_run_updates": main_run_updates,
                "duration": time.perf_counter() - time_ref,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                "parameters": {
                    "connector": {
                        "trainable": connector_trainable,
                        "total": connector_total,
                    },
                    "llm": {
                        "trainable": llm_trainable,
                        "total": llm_total,
                    },
                }
            })
            total_loss_per_opt = 0.0
            time_ref = time.perf_counter()
            
            if main_run_updates % 250 == 0:
                ckpt_path_step = ckpt_path / f"step_{global_step}.pt"
                save_ckpt(ckpt_path_step, step, global_step, connector, opt, scheduler, scaler, llm, stage, logs)

        if global_step >= total_steps:
            break
            
    progress_bar.close()
    print(f"Stage {stage} completed successfully.")
