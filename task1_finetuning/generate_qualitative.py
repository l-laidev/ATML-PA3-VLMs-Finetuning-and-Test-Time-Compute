import os
import yaml
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any

from task1_finetuning.utils import set_seed


def load_qualitative_prompts(yaml_path: str = "data/first_plot_questions.yaml") -> List[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompts = []
    for entry in data:
        format_ = "template" if entry["id"].endswith("_template") else "json" if entry["id"].endswith("_json") else "plain"
        question = entry["id"].replace(f"_{format_}", "")
        prompt = entry["paraphrases"][0]
        prompts.append({"prompt": prompt, "question": question, "format": format_})
        
    return prompts


def generate_responses_for_condition(
    model_path: str,
    condition_name: str,
    prompts: List[dict],
    output_dir: str = "results/task1/qualitative",
    device: str = "cuda"
):
    print(f"\n[Condition: {condition_name}] Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    results = {}
    
    for idx, entry in enumerate(prompts):
        prompt, question, format_ = entry["prompt"], entry["question"], entry["format"]
        formatted_prompt = tokenizer.apply_chat_template(
            [{"content": prompt, "role": "user"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Set to False to skip <think> blocks
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if question not in results:
            results[question] = {}
        if format_ not in results[question]:
            results[question][format_] = {}
        
        results[question][format_] = response_text
        print(f"  Generated response {idx+1}/{len(prompts)}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{condition_name}_responses.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(results)} responses for {condition_name} to {output_file}")


def main():
    set_seed(6304)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluation_conditions = {
        "baseline": "Qwen/Qwen3-0.6B",
        "full_ft_midpoint": "./checkpoints/task1/full_ft/checkpoint_update_625_examples_20000",
        "full_ft_final": "./checkpoints/task1/full_ft/checkpoint_update_1250_examples_40000",
        "l2_sp_final": "./checkpoints/task1/l2_sp/checkpoint_update_1250_examples_40000",
        "lora_r4_final": "./checkpoints/task1/lora_r4/merged_model",
        "lora_r16_final": "./checkpoints/task1/lora_r16/merged_model",
        "oplora_r16_final": "./checkpoints/task1/oplora_r16/merged_model"
    }

    prompts = load_qualitative_prompts()
    print(f"Loaded {len(prompts)} qualitative prompt variations.")

    for cond_name, path in evaluation_conditions.items():
        try:
            generate_responses_for_condition(
                model_path=path,
                condition_name=cond_name,
                prompts=prompts,
                device=device
            )
        except Exception as e:
            print(f"Skipping condition '{cond_name}' due to missing checkpoint or error: {e}")


if __name__ == "__main__":
    main()