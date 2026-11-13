import json
import time
import tqdm
from pathlib import Path

import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from task3_test_time_compute.common import create_splits, load_math, preview_ith_row
from task3_test_time_compute.main import math_verify_check, normalize_math


def load_model():
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        device_map="auto",
    )
    
    return model, tokenizer

def generate(prompt, tokenizer, model, max_new_tokens, to_refine=None):
    formatting_instruction = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
    refinement_instruction = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."
    
    stats = {}

    user_prompt = f"{prompt}\n\n{formatting_instruction}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt}
    ]
    
    if to_refine is not None:
        messages.extend([
            {"role": "assistant", "content": to_refine},
            {"role": "user", "content": refinement_instruction},
        ])
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    stats["input_tokens"] = inputs.input_ids

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    stats["is_truncated"] = (outputs[0][-1].item() != tokenizer.eos_token_id)
    stats["runtime"] = time.time() - start

    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    stats["output_tokens"] = generated_tokens

    return response, stats

def extract_ans(text):
    target = r"\boxed{"
    idx = text.rfind(target)  # last boxed answer
    if idx == -1:
        return None
    
    # after the opening { of \boxed{
    start_idx = idx + len(target)
    
    depth = 1
    i = start_idx
    in_escape = False
    
    while i < len(text):
        char = text[i]
        
        if in_escape:
            in_escape = False
        elif char == '\\':
            in_escape = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                # found the matching closing brace
                extracted = text[start_idx:i].strip()
                return extracted if extracted else None
        
        i += 1
    
    # unclosed
    return None

def evaluate_live(df: pd.DataFrame, splits: dict, out_base: str, split: str):
    out_base = Path(out_base) if not isinstance(out_base, Path) else out_base
    out_base = out_base / split
    out_base.mkdir(exist_ok=True, parents=True)
    
    model, tokenizer = load_model()
    for cur_ds, math_splits in splits.items():
        cur_df = df[df["sample_id"].isin(math_splits[split])]
        unique_sids = cur_df["sample_id"].unique()
        
        print(f"\n--- Live Generation for Dataset ({split.title()}): {cur_ds} ({len(unique_sids)} questions) ---") 
        
        initial = {
            "runtime": 0.0,
            "accuracy": 0.0,
            "truncation_rate": 0.0,
            "cost": 0.0,
        }
        totals = {strat: {**initial} for strat in ["single", "parallel", "refine"]}
        totals["refine"]["wrong_to_correct"] = 0.0
        totals["refine"]["correct_to_wrong"] = 0.0
        
        evidences = []
        
        for i, sid in tqdm.tqdm(enumerate(unique_sids), desc="Generating", total=len(unique_sids)):
            q_data = df[df["sample_id"] == sid]
            gold_ans = q_data["gold_answer"].iloc[0]
            problem = q_data["question_text"].iloc[0]
            
            
            single, stats_single = generate(problem, tokenizer, model, max_new_tokens=3_072)
            
            parallel = []
            stats_parallel = []
            for _ in range(3):
                res, stats_res = generate(problem, tokenizer, model, max_new_tokens=1024)
                parallel.append(res)
                stats_parallel.append(stats_res)
            
            refine, stats_refine = generate(problem, tokenizer, model, max_new_tokens=1024, to_refine=parallel[0])
            
            stats_single["is_correct"] = math_verify_check(gold_ans, normalize_math(extract_ans(single)))
            for i in range(3):
                stats_parallel[i]["is_correct"] = math_verify_check(gold_ans, normalize_math(extract_ans(parallel[i])))
            stats_refine["is_correct"] = math_verify_check(gold_ans, normalize_math(extract_ans(refine)))
            
            totals["single"]["runtime"] += stats_single["runtime"]
            totals["parallel"]["runtime"] += sum(p["runtime"] for p in stats_parallel)
            totals["refine"]["runtime"] += (stats_parallel[0]["runtime"] + stats_refine["runtime"])
            
            totals["single"]["truncation_rate"] += int(stats_single["is_truncated"])
            totals["parallel"]["truncation_rate"] += int(any(p["is_truncated"] == True for p in stats_parallel))
            totals["refine"]["truncation_rate"] += int(stats_parallel[0]["is_truncated"] or stats_refine["is_truncated"])
            
            totals["single"]["cost"] += len(stats_single["output_tokens"])
            totals["parallel"]["cost"] += sum(len(p["output_tokens"]) for p in stats_parallel)
            totals["refine"]["cost"] += (
                len(stats_refine["output_tokens"])
                + (len(stats_refine["input_tokens"]) - len(stats_parallel[0]["input_tokens"]))  # token cost for the refinement prompt and also the initial response
            )
            
            totals["single"]["accuracy"] += int(stats_single["is_correct"])
            totals["parallel"]["accuracy"] += int(any(p["is_correct"] for p in stats_parallel))
            totals["refine"]["accuracy"] += int(stats_refine["is_correct"])
            
            # correct-to-wrong
            if stats_parallel[0]["is_correct"] and not stats_refine["is_correct"]:
                totals["refine"]["correct_to_wrong"] += 1
            # wrong-to-correct
            elif not stats_parallel[0]["is_correct"] and stats_refine["is_correct"]:
                totals["refine"]["wrong_to_correct"] += 1
            
            # TODO: peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            evidences.append({
                "prompt": problem,
                "responses": {
                    "single": {"response": single, "length": len(stats_single["output_tokens"]), "is_truncated": stats_single["is_truncated"], "runtime": stats_single["runtime"]},
                    "parallel": [
                        {"response": parallel[i], "length": len(stats_parallel[i]["output_tokens"]),
                         "is_truncated": stats_parallel[i]["is_truncated"], "runtime": stats_parallel[i]["runtime"]}
                        for i in range(3)],
                    "refine": {"response": refine, "length": totals["refine"]["cost"], "is_truncated": stats_refine["is_truncated"], "runtime": stats_refine["runtime"]},
                },
            })
    
        with open(out_base / "overview.json", "w") as f:
            f.write(json.dumps(totals))
        with open(out_base / "evidence.json", "w") as f:
            f.write(json.dumps(evidences))


def main():
    pd.set_option('display.max_colwidth', None)
    ds_folder = "qwen25-math"
    BASE = Path("data/task3/ATML-PA3-Data") / ds_folder
    
    output_dir = BASE / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_math(BASE)
    print(f"Shape: {df.shape}")
    
    df["normalized_answer"] = df["extracted_answer"].apply(normalize_math)
    df["gold_answer"] = df["gold_answer"].apply(normalize_math)
    
    preview_ith_row(df, ith=0)
    
    splits = create_splits(df, seed=6304)
    split_file = output_dir / "split_ids.json"
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved split IDs to {split_file}")
    
    RESULTS_BASE = Path("results/task3") / "refinement"
    RESULTS_BASE.mkdir(exist_ok=True, parents=True)
    
    evaluate_live(df, splits, RESULTS_BASE / "live_gen", split="live_gen")


if __name__ == "__main__":
    main()
