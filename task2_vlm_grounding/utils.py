import re
import time

import torch
import json
from pathlib import Path

import os
import random
import numpy as np
from transformers import set_seed as set_seed_hf
from peft import PeftModel


def set_seed(seed: int = 6304):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_seed_hf(seed)

def append_jsonl(path: str | Path, data: dict):
    p = path if isinstance(path, Path) else Path(path)
    
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    
    with open(p, "a") as f:
        f.write(json.dumps(data) + '\n')

def save_ckpt(ckpt_path, step, global_step, connector, opt, scheduler, scaler, llm, stage, logs):
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save({
        "step": step,
        "global_step": global_step,
        "connector_state": connector.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": rng_state,
    }, ckpt_path)

    with open(Path(ckpt_path).with_suffix(".json"), "w") as f:
        json.dump(logs, f)
    
    if stage != 1:
        llm.save_pretrained(Path(ckpt_path).parent / f"adapter_step_{global_step}")

def load_ckpt(ckpt_path: Path, connector, opt, scheduler, scaler, llm, stage):
    base_ckpt = ckpt_path
    if ckpt_path.is_file():
        checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    else:
        latest = list(sorted(filter(lambda x: x.endswith(".pt"), os.listdir(ckpt_path)), key=lambda x: (len(x), x)))[-1]
        print(latest)
        ckpt_path = ckpt_path / latest
        checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)

    connector.load_state_dict(checkpoint["connector_state"])
    opt.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    
    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    
    if "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["torch_cuda"]])
    
    with open(ckpt_path.with_suffix(".json"), "r") as f:
        logs = json.load(f)
    
    step = checkpoint["step"]
    global_step = checkpoint["global_step"]
    
    if stage != 1:
        llm = PeftModel.from_pretrained(llm, base_ckpt / f"adapter_step_{global_step}")
    
    
    return step, global_step, logs

def normalize_answer(answer):
    if not answer:
        return ""
    
    answer = answer.lower()
    answer = re.sub(r'\b(a|an|the)\b', '', answer)
    
    num_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
    }
    for word, digit in num_words.items():
        answer = re.sub(r'\b' + word + r'\b', digit, answer)
        
    contractions = {"dont": "don't", "cant": "can't", "wont": "won't", "isnt": "isn't"}
    for missing, correct in contractions.items():
        answer = re.sub(r'\b' + missing + r'\b', correct, answer)
        
    answer = re.sub(r'\.(?!\d)', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer

def prepare_batch(llm, tokenizer, prompts, targets, visual_embeds, is_caption=False):
    embed_tokens = llm.get_input_embeddings() if not hasattr(llm, "base_model") else llm.base_model.get_input_embeddings()
    
    full_prompt = list(map(lambda prompt: f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", prompts))
    prompt_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False, enable_thinking=False, padding=True, padding_side="left")
    prompt_masks = prompt_inputs.attention_mask.to(llm.device)
    prompt_ids = prompt_inputs.input_ids.to(llm.device)
    # question cap
    prompt_masks = prompt_masks[:, :prompt_ids.shape[1] if is_caption else 80]
    prompt_ids = prompt_ids[:, :prompt_ids.shape[1] if is_caption else 80]
    prompt_embeds = embed_tokens(prompt_ids.to(llm.device))
    
    if targets is not None:
        full_target = list(map(lambda target: f"{target}<|im_end|>", targets))
        target_inputs = tokenizer(full_target, return_tensors="pt", add_special_tokens=False, enable_thinking=False, padding=True)
        target_masks = target_inputs.attention_mask.to(llm.device)
        target_ids = target_inputs.input_ids.to(llm.device)
        # short-answer cap
        target_masks = target_masks[:, :64 if is_caption else 16]
        target_ids = target_ids[:, :64 if is_caption else 16]
        target_embeds = embed_tokens(target_ids.to(llm.device))
    else:
        target_masks = None
        target_ids = None
        target_embeds = None

    current_visual_emb = None
    if visual_embeds is not None:
        current_visual_emb = visual_embeds.to(prompt_embeds.dtype).squeeze(dim=1)  # [B, 1, 196, 1024] -> [B, 196, 1024]

    # 1st three prompt embs are from the im_start, "user" role and \n tokens
    prefix_len = 3
    combined_embeds = [prompt_embeds[:, :prefix_len]]
    
    if current_visual_emb is not None:
        combined_embeds.append(current_visual_emb)

    combined_embeds.append(prompt_embeds[:, prefix_len:])
    if target_embeds is not None:
        combined_embeds.append(target_embeds)

    combined_embeds = torch.cat(combined_embeds, dim=1)

    batch_size = prompt_embeds.shape[0]
    prompt_len = prompt_embeds.shape[1]
    visual_len = current_visual_emb.shape[1] if current_visual_emb is not None else 0
    total_prompt_len = prompt_len + visual_len

    prompt_labels = torch.full((batch_size, total_prompt_len), -100, dtype=torch.long, device=llm.device)
    
    if target_ids is not None:
        combined_labels = torch.cat([prompt_labels, target_ids], dim=1)
        # mask out padding tokens
        if target_masks is not None:
            combined_labels[:, total_prompt_len:][target_masks == 0] = -100
    else:
        combined_labels = prompt_labels

    visual_masks = torch.ones((batch_size, visual_len), dtype=torch.long, device=llm.device) if visual_len > 0 else None
    mask_list = [prompt_masks[:, :prefix_len]]
    if visual_masks is not None:
        mask_list.append(visual_masks)
    mask_list.append(prompt_masks[:, prefix_len:])
    
    if target_masks is not None:
        mask_list.append(target_masks)
        
    attention_mask = torch.cat(mask_list, dim=1)

    return combined_embeds, attention_mask, combined_labels

def process_batch_loss(llm, tokenizer, prompts, targets, visual_embeds, is_caption=False):
    combined_embeds, attention_mask, combined_labels = prepare_batch(llm, tokenizer, prompts, targets, visual_embeds, is_caption)
    outputs = llm(inputs_embeds=combined_embeds, attention_mask=attention_mask, labels=combined_labels)
    loss = outputs.loss
    return loss

def process_batch_generate(llm, tokenizer, prompts, visual_embeds):
    combined_embeds, attention_mask, _ = prepare_batch(llm, tokenizer, prompts, None, visual_embeds)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    duration = time.perf_counter() - start
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated, duration
