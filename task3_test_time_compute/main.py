from argparse import ArgumentParser
import json
import os
import re
import tqdm
from pathlib import Path

import numpy as np
import pandas as pd

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException

from task3_test_time_compute.common import compare_methods, compute_rollout_metrics, create_splits, load_math, preview_ith_row, process_adaptive_stopping, rank_weighted_consensus


def normalize_math(ans: str) -> str:
    if ans is None:
        return "NO_ANSWER"
    
    return re.sub(r'\s', '', ans)
    # if ans is None or str(ans).strip() == "" or str(ans).lower() == "nan":
    #     return "NO_ANSWER"
    
    # if isinstance(ans, (list, tuple)):
    #     ans = ans[0] if len(ans) > 0 else "NO_ANSWER"
        
    # ans_str = str(ans).strip()
    
    # try:
    #     parsed = parse(ans_str)
    #     if parsed is not None:
    #         if isinstance(parsed, (list, tuple)):
    #             parsed = parsed[0] if len(parsed) > 0 else ans_str
    #         return str(parsed)
    # except Exception:
    #     pass
    
    # cleaned = ans_str.lower()
    # cleaned = cleaned.replace("\\$", "").replace("$", "").replace("\\text{", "").replace("}", "")
    # return cleaned.strip()

def math_verify_check(gold: str, ans: str) -> bool:
    if ans == "NO_ANSWER":
        return False
    
    gold = rf"\boxed{{{gold}}}"
    ans = rf"\boxed{{{ans}}}"
    
    g_parsed = None
    a_parsed = None
    try:
        g_parsed = parse(gold, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()], parsing_timeout=10, raise_on_error=True)
        a_parsed = parse(ans, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()], parsing_timeout=10, raise_on_error=True)
        res = bool(verify(g_parsed, a_parsed, timeout_seconds=10, raise_on_error=True))
    except TimeoutException as e:
        # print(f"\nException {type(e)} at {gold=}, {g_parsed=}, {ans=}, {a_parsed=}")
        return False
    
    return res

def evaluate_offline_budgets(df: pd.DataFrame, splits: dict, out_base: str, split: str):
    out_base = Path(out_base) if not isinstance(out_base, Path) else out_base
    out_base = out_base / split
    out_base.mkdir(exist_ok=True, parents=True)
    
    budgets = [1, 2, 4, 8, 16, 32]
    N_perm = 10
    
    for cur_ds, math_splits in splits.items():
        cur_df = df[df["sample_id"].isin(math_splits[split])]
        unique_sids = cur_df["sample_id"].unique()
        
        print(f"\n--- Offline Evaluation for Dataset ({split.title()}): {cur_ds} ({len(unique_sids)} questions) ---")
        
        for budget in tqdm.tqdm(budgets, desc="Budget", total=len(budgets)):
            metrics, _ = compare_methods(cur_df, N_perm, budget, verifier_fn=math_verify_check, adaptive_stop_fn=None)
            print(json.dumps(metrics))
            with open(out_base / f"budget-{budget}.json", "w") as f:
                f.write(json.dumps(metrics))

def evaluate_adaptive_stopping(df: pd.DataFrame, splits: dict, out_base: str, split: str):
    out_base = Path(out_base) if not isinstance(out_base, Path) else out_base
    out_base = out_base / split
    out_base.mkdir(exist_ok=True, parents=True)
    
    budgets = [32]
    N_perm = 10
    
    for cur_ds, math_splits in splits.items():
        cur_df = df[df["sample_id"].isin(math_splits[split])]
        unique_sids = cur_df["sample_id"].unique()
        
        print(f"\n--- Adaptive Stoping Evaluation for Dataset ({split.title()}): {cur_ds} ({len(unique_sids)} questions) ---")

        policies = ["agreement", "quality_aware"]
        taus = [0.5, 0.625, 0.75, 0.875, 1.0]
        gammas = [0, 0.1, 0.25, 0.5]
        
        for policy in (t0 := tqdm.tqdm(policies, total=len(policies))):
            t0.set_description_str(f"Policy={policy}")
            
            for tau in (t1 := tqdm.tqdm(taus, leave=False, total=len(taus))):
                t1.set_description_str(f"Tau={tau}")
                
                for gamma in (t2 := tqdm.tqdm(gammas if policy == "quality_aware" else [-1], leave=False, total=len(gammas))):
                    t2.set_description_str(f"Gamma={gamma}")
                    
                    for budget in tqdm.tqdm(budgets, desc="Budget", leave=False):
                        metrics_path = out_base / f"budget-{budget}-adaptive-{policy}-{tau}-{gamma}-metrics.json"
                        if metrics_path.exists():
                            continue
                        
                        adapt_fn = lambda q_data, order: process_adaptive_stopping(q_data, order,
                                                                                   tau=tau, gamma=gamma, policy_type=policy,
                                                                                   score_col="K_i" if "mathvista" not in cur_ds else "Z_i")
                        metrics, adaptive_stopping_insights = compare_methods(cur_df, N_perm, budget, verifier_fn=math_verify_check, adaptive_stop_fn=adapt_fn)
                        
                        with open(metrics_path, "w") as f:
                            f.write(json.dumps(metrics))
                        with open(out_base / f"budget-{budget}-adaptive-{policy}-{tau}-{gamma}-insights.json", "w") as f:
                            f.write(json.dumps(adaptive_stopping_insights))

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["math", "mathvista"])
    args = parser.parse_args()
    
    pd.set_option('display.max_colwidth', None)
    ds_folder = "qwen25-math" if args.dataset == "math" else "qwen25vl-mathvista"
    BASE = Path("data/task3/ATML-PA3-Data") / ds_folder
    
    output_dir = BASE / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_math(BASE)
    print(f"Shape: {df.shape}")
    
    df["normalized_answer"] = df["extracted_answer"].apply(normalize_math)
    df["gold_answer"] = df["gold_answer"].apply(normalize_math)
    
    # for mathvista
    if "n_generated_text_tokens" not in df.columns:
        df["n_generated_text_tokens"] = len(df["generated_token_ids"])
    
    metrics_records = [compute_rollout_metrics(row) for _, row in df.iterrows()]
    metrics_df = pd.DataFrame(metrics_records, index=df.index)
    for col in metrics_df.columns:
        df[col] = metrics_df[col]
    
    preview_ith_row(df, ith=0)
    
    splits = create_splits(df, seed=6304)
    split_file = output_dir / "split_ids.json"
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved split IDs to {split_file}")
    
    RESULTS_BASE = Path("results/task3") / args.dataset
    RESULTS_BASE.mkdir(exist_ok=True, parents=True)
    
    if not (RESULTS_BASE / "main").exists():
        evaluate_offline_budgets(df, splits, RESULTS_BASE / "main", split="dev")
    evaluate_adaptive_stopping(df, splits, RESULTS_BASE / "adaptive", split="dev")
    
    # evaluate_offline_budgets(df, splits, RESULTS_BASE / "main", split="eval")
    # evaluate_adaptive_stopping(df, splits, RESULTS_BASE / "adaptive", split="eval")


if __name__ == "__main__":
    main()
