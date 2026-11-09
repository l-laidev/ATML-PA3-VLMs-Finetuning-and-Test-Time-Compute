import os
import re
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

def normalize_text(text: str) -> str:
    """Normalizes text for matching GSM8K test questions."""
    return re.sub(r'\s+', ' ', text).strip().lower()

def tokenize(tokenizer, query, response):
    prompt = tokenizer.apply_chat_template(
        [{"content": query, "role": "user"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Set to False to skip <think> blocks
    )
    response_text = f"{response}<|im_end|>"


    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    
    input_ids = prompt_ids + response_ids
    
    return prompt_ids, response_ids, input_ids

def prepare_and_cache_dataset(
    tokenizer: PreTrainedTokenizer,
    cache_dir: str = "data/metamath_tokenized",
    max_length: int = 2048
) -> Dataset:
    """
    Loads MetaMathQA and GSM8K test set, filters per assignment criteria:
      - Exclude empty questions or responses.
      - Exclude formatted length > 2,048 tokens.
      - Exclude queries matching GSM8K test questions (after normalization).
      - Retain first 40,000 eligible examples in source order without shuffling.
    """
    if os.path.exists(cache_dir):
        print(f"Loading pre-tokenized dataset from cache: {cache_dir}")
        return Dataset.load_from_disk(cache_dir)

    print("Downloading/loading MetaMathQA and GSM8K test set for filtering...")
    raw_dataset = load_dataset("meta-math/MetaMathQA", split="train")
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_questions = {normalize_text(q) for q in gsm8k_test["question"]}

    eligible_examples = []
    n_empty = 0
    n_overlapped = 0
    n_exceed_length = 0
    for item in raw_dataset:
        query = item["query"]
        response = item["response"]
        
        if not query.strip() or not response.strip():
            n_empty += 1
            continue
            
        if normalize_text(query) in gsm8k_questions or normalize_text(item["original_question"]) in gsm8k_questions:
            n_overlapped += 1
            continue

        _, _, input_ids = tokenize(tokenizer, query, response)
        if len(input_ids) > 2048:
            n_exceed_length += 1
            continue
            
        eligible_examples.append({"query": query, "response": response})
        if len(eligible_examples) >= 40_000:
            break

    eligible_examples = eligible_examples[:40_000]
    print(f"Collected {len(eligible_examples)} eligible examples. {n_empty} empty query/response, {n_exceed_length} exceeded 2048 tokens, and {n_overlapped} overlapped with GSM8K")
    print("Tokenizing...")

    def tokenize_function(batch):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for query, response in zip(batch["query"], batch["response"]):
            prompt_ids, response_ids, input_ids = tokenize(tokenizer, query, response)
            # Mask prompt tokens with -100, compute cross-entropy only on assistant response and end-of-turn
            labels = [-100] * len(prompt_ids) + response_ids
            
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                
            attention_mask = [1] * len(input_ids)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list
        }

    temp_dataset = Dataset.from_list(eligible_examples)
    tokenized_dataset = temp_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["query", "response"],
        desc="Tokenizing MetaMathQA"
    )

    print(f"Saving tokenized dataset to {cache_dir}...")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    tokenized_dataset.save_to_disk(cache_dir)
    
    return tokenized_dataset