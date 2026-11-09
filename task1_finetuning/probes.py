import os
import json
import random
from datasets import load_dataset

def get_or_create_probe_set(tokenizer, cache_path: str = "data/activation_probe_set.json", seed: int = 6304) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    rng = random.Random(seed)

    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_indices = list(range(len(gsm8k_test)))
    rng.shuffle(gsm8k_indices)
    selected_gsm8k_idx = gsm8k_indices[:32]

    math_probes = []
    for idx in selected_gsm8k_idx:
        item = gsm8k_test[idx]
        question = item["question"]
        prompt = tokenizer.apply_chat_template(
            [{"content": question, "role": "user"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Set to False to skip <think> blocks
        )
        math_probes.append({"id": f"gsm8k_{idx}", "prompt": prompt})

    hellaswag_val = load_dataset("Rowan/hellaswag", split="validation")
    hellaswag_indices = list(range(len(hellaswag_val)))
    rng.shuffle(hellaswag_indices)
    selected_hellaswag_idx = hellaswag_indices[:32]

    non_math_probes = []
    for idx in selected_hellaswag_idx:
        item = hellaswag_val[idx]
        context = item["ctx"]
        prompt = tokenizer.apply_chat_template(
            [{"content": context, "role": "user"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Set to False to skip <think> blocks
        )
        non_math_probes.append({"id": f"hellaswag_{idx}", "prompt": prompt})

    probe_payload = {"seed": seed, "math_probes": math_probes, "non_math_probes": non_math_probes}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(probe_payload, f, indent=2, ensure_ascii=False)

    return probe_payload