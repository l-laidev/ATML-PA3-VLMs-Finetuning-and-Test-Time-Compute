import numpy as np
import pandas as pd

from pathlib import Path

import tqdm


def load_math(base: Path) -> pd.DataFrame:
    generations = pd.read_parquet(base / "generations.parquet")
    metrics = pd.read_parquet(base / "probability_metrics.parquet")
    
    attn_metrics = None
    if (base / "attention_metrics.parquet").exists():
        print("Found attention_metrics file")
        attn_metrics = pd.read_parquet(base / "attention_metrics.parquet")
    
    df = generations.merge(metrics, on=["dataset", "sample_id", "rollout_id"])
    
    if attn_metrics is not None:
        df = df.merge(attn_metrics, on=["dataset", "sample_id", "rollout_id"])
    
    return df

def create_splits(df: pd.DataFrame, seed: int = 6304):
    datasets = df["dataset"].unique()
    splits = {}
    rng = np.random.default_rng(seed)

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        question_ids = sorted(ds_df["sample_id"].unique())
        permuted_ids = rng.permutation(question_ids)
        
        dev_ids = permuted_ids[:200].tolist()
        eval_ids = permuted_ids[200:].tolist()
        live_gen_ids = eval_ids[:50]
        
        splits[ds] = {
            "dev": dev_ids,
            "eval": eval_ids,
            "live_gen": live_gen_ids
        }
    return splits

def preview_ith_row(df, ith=0):
    for col in df.columns:
        val = df[col].iloc[ith]
        
        if isinstance(val, list):
            val = f"[{val[0]}], N={len(val)}"
        elif isinstance(val, str):
            val = val[:50] + "..."
        
        print(f"- {col}: {val}")

def compute_rollout_metrics(row) -> dict:
    logp_list = row["logp"]
    kl_list = row["kl_u_p"]
    
    L_i = float(np.mean(logp_list))
    K_i = float(np.mean(kl_list))
    
    metrics = {"L_i": L_i, "K_i": K_i}
    
    if "img_attn_total" in row:
        A_img_it = np.array(row["img_attn_total"])
        A_prmpt_it = np.array(row["prompt_text_attn_total"])
        A_gen_it = np.array(row["generated_text_attn_total"])
            
        v_it = A_img_it / (A_img_it + A_prmpt_it + A_gen_it)
        V_i = float(np.mean(v_it))
        metrics["V_i"] = V_i
    
    return metrics

def visconf_inplace(candidate_set):
    pooled_kl = []
    for i, row in candidate_set.iterrows():
        pooled_kl.extend(row["kl_u_p"])

    percentile_5th = np.percentile(pooled_kl, 5, method="linear")
    percentile_95th = np.percentile(pooled_kl, 95, method="linear")
    
    def _visconf(row):
        k_it = row["kl_u_p"]
        c_it = (k_it - percentile_5th) / max(percentile_95th - percentile_5th, 1e-8)
        c_it = np.clip(c_it, 0.0, 1.0)
        C_i = float(np.mean(c_it))
        return C_i
    
    candidate_set["C_i"] = candidate_set.apply(_visconf, axis=1)
    candidate_set["Z_i"] = candidate_set["C_i"] * candidate_set["V_i"]

def rank_weighted_consensus(selected_df: pd.DataFrame, score_col: str, skip_if_one: bool = True) -> str:
    N = len(selected_df)
    if N <= 1 and skip_if_one:
        return selected_df.iloc[0]["normalized_answer"] if N == 1 else "NO_ANSWER", None
        
    sorted_df = selected_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    
    ans_weights = {}
    for rank_idx, row in sorted_df.iterrows():
        r_i = rank_idx + 1  # 1-based
        w_i = (N - r_i) / (N - 1)
        ans = row["normalized_answer"]
        ans_weights[ans] = ans_weights.get(ans, 0.0) + w_i
    
    sorted_weights = sorted(ans_weights.items(), key=lambda x: x[1], reverse=True)
    best_ans = sorted_weights[0][0]
    
    best_weight = sorted_weights[0][1]
    runner_up_weight = sorted_weights[1][1] if len(sorted_weights) > 1 else 0.0
    total_weight = sum(ans_weights.values())
    rwc_margin = (best_weight - runner_up_weight) / total_weight
    
    return best_ans, rwc_margin


def paired_confidence_intervals(meth1_data, meth2_data, n_resamples=1_000, seed=6304):
    rng = np.random.default_rng(seed)
    ids = list(range(len(meth1_data)))
    diffs = []
    for _ in range(n_resamples):
        sampled = rng.choice(ids, size=len(meth1_data), replace=True)
        diff = meth1_data[sampled].mean() - meth2_data[sampled].mean()
        diffs.append(diff)
    return {
        "mean": np.mean(diffs),
        "lower": np.percentile(diffs, 2.5),
        "upper": np.percentile(diffs, 97.5),
    }

def process_one(selected, gold_ans, verifier_fn, scores):
    is_correct_list = [verifier_fn(gold_ans, ans) for ans in selected["normalized_answer"]]
    oracle = 1.0 if any(is_correct_list) else 0.0
    oracle_exists = (oracle == 1.0)
    
    totals = {k: 0.0 for k in scores}
    sel_fails = {k: 0.0 for k in scores}
    sam_fails = {k: 0.0 for k in scores}
    sc_correctness = 0.0
    for k in scores:
        val = None
        if k.startswith("assoc-"): continue  # associations are for 32 budget
        
        if k == "cost":
            val = selected["n_generated_text_tokens"].sum().tolist()
        elif k == "distinct":
            val = len(set(selected["normalized_answer"].tolist()))
        elif k == "oracle":
            val = oracle
        elif k == "fa":
            val = 1.0 if verifier_fn(gold_ans, selected.iloc[0]["normalized_answer"]) else 0.0
        elif k == "sc":
            vote_counts = selected["normalized_answer"].value_counts()
            sc_winner = vote_counts.idxmax()
            val = 1.0 if verifier_fn(gold_ans, sc_winner) else 0.0
            sc_correctness = val
        elif k == "logp":
            logp_winner = selected[selected["L_i"] == selected["L_i"].max()].sort_values(by="rollout_id").iloc[0]["normalized_answer"]
            val = 1.0 if verifier_fn(gold_ans, logp_winner) else 0.0
        elif k == "scert":
            scert_winner = selected[selected["K_i"] == selected["K_i"].max()].sort_values(by="rollout_id").iloc[0]["normalized_answer"]
            val = 1.0 if verifier_fn(gold_ans, scert_winner) else 0.0
        elif k == "rwc":
            rwc_winner, rwc_margin = rank_weighted_consensus(selected, score_col="K_i")
            val = 1.0 if verifier_fn(gold_ans, rwc_winner) else 0.0
        elif k == "vis":
            vis_winner = selected[selected["V_i"] == selected["V_i"].max()].sort_values(by="rollout_id").iloc[0]["normalized_answer"]
            val = 1.0 if verifier_fn(gold_ans, vis_winner) else 0.0
        elif k == "visconf":
            visconf_winner = selected[selected["Z_i"] == selected["Z_i"].max()].sort_values(by="rollout_id").iloc[0]["normalized_answer"]
            val = 1.0 if verifier_fn(gold_ans, visconf_winner) else 0.0
        elif k == "vis_rwc":
            vis_rwc_winner, vis_rwc_margin = rank_weighted_consensus(selected, score_col="V_i")
            val = 1.0 if verifier_fn(gold_ans, vis_rwc_winner) else 0.0
        elif k == "visconf_rwc":
            visconf_rwc_winner, visconf_rwc_margin = rank_weighted_consensus(selected, score_col="Z_i")
            val = 1.0 if verifier_fn(gold_ans, visconf_rwc_winner) else 0.0
        
        if k in ["fa", "sc", "logp", "scert", "rwc", "vis", "visconf", "vis_rwc", "visconf_rwc"]:
            if val == 0.0:
                if oracle_exists:
                    sel_fails[k] += 1
                else:
                    sam_fails[k] += 1
        
        totals[k] += val
    
    return totals, sel_fails, sam_fails, sc_correctness

def process_adaptive_stopping(q_data: pd.DataFrame, rollout_order: np.ndarray, tau: float, gamma: float, policy_type: str, score_col: str) -> dict:
    has_attn = ("img_attn_total" in q_data.columns)
    checkpoints = [4, 8, 16, 32]
    stopped_at = 32
    final_answer = "NO_ANSWER"
    tokens_consumed = 0
    
    for n in checkpoints:
        revealed_rids = rollout_order[:n]
        selected = q_data[q_data["rollout_id"].isin(revealed_rids)].copy()
        tokens_consumed = selected["n_generated_text_tokens"].sum()
        
        if len(selected) == 0:
            continue
        
        # self-consistency selection
        vote_counts = selected["normalized_answer"].value_counts(normalize=True)
        top_answer = vote_counts.idxmax()
        top_fraction = vote_counts.max()
        
        agreement_met = (top_answer != "NO_ANSWER") and (top_fraction >= tau)
        
        if n == 32:
            stopped_at = 32
            final_answer = top_answer
            break
            
        if policy_type == "agreement":
            if agreement_met:
                stopped_at = n
                final_answer = top_answer
                break
                
        elif policy_type == "quality_aware":
            if has_attn:
                visconf_inplace(selected)
                
            rwc_winner, rwc_margin = rank_weighted_consensus(selected, score_col=score_col, skip_if_one=False)
            
            sc_rwc_agree = (top_answer == rwc_winner)
            margin_met = (rwc_margin >= gamma)
            
            if agreement_met and sc_rwc_agree and margin_met:
                stopped_at = n
                final_answer = top_answer
                break
                
    return {
        "stopped_at": stopped_at,
        "final_answer": final_answer,
        "tokens_consumed": tokens_consumed
    }

def compare_methods(df, N_perm, budget, verifier_fn, adaptive_stop_fn=None):
    has_attn = ("img_attn_total" in df.columns)
    
    unique_sids = df["sample_id"].unique()
    totals = {k: 0.0 for k in ["fa", "sc", "oracle", "cost", "logp", "scert", "rwc", "distinct"]}
    sel_fails = {k: 0.0 for k in ["fa", "sc", "logp", "scert", "rwc"]}
    sam_fails = {k: 0.0 for k in sel_fails.keys()}
    
    adaptive_totals = {k: 0.0 for k in ["cost", "sc", "stopped_at"]}
    adaptive_stopping_dst = []
    incorrect_stopping = []
    
    if has_attn:
        for k in ["vis", "visconf"]:
            for suffix in ["", "_rwc"]:
                totals[k+suffix] = 0.0
                sel_fails[k+suffix] = 0.0
                sam_fails[k+suffix] = 0.0
    
    for i, sid in tqdm.tqdm(enumerate(unique_sids), desc="Sample", total=len(unique_sids)):
        q_data = df[df["sample_id"] == sid]
        gold_ans = q_data["gold_answer"].iloc[0]
        rollout_ids = q_data["rollout_id"].values
        rng = np.random.default_rng(6304)
        orderings = [rng.permutation(rollout_ids) for _ in range(N_perm)]
        
        perm_totals = {k: 0.0 for k in totals.keys()}
        perm_sel_fails = {k: 0 for k in sel_fails.keys()}
        perm_sam_fails = {k: 0 for k in sam_fails.keys()}
        perm_adaptive_totals = {k: 0.0 for k in adaptive_totals.keys()}
        
        for j in tqdm.trange(N_perm, leave=False, desc="Permutation"):
            shuffled_rids = orderings[j]
            selected_rids = shuffled_rids[:budget]
            selected = q_data[q_data["rollout_id"].isin(selected_rids)].copy()
            
            if has_attn:
                visconf_inplace(selected)
            
            cur_totals, cur_sel_fails, cur_sam_fails, sc_correctness = process_one(selected, gold_ans, verifier_fn, perm_totals.keys())
            
            for k, v in cur_totals.items():
                perm_totals[k] += v
            
            for k, v in cur_sel_fails.items():
                if k in perm_sel_fails:
                    perm_sel_fails[k] += v
            
            for k, v in cur_sam_fails.items():
                if k in perm_sam_fails:
                    perm_sam_fails[k] += v

            if adaptive_stop_fn is not None:
                order = orderings[j]
                adaptive_ans = adaptive_stop_fn(q_data, order)
                adaptive_correctness = 1.0 if verifier_fn(gold_ans, adaptive_ans["final_answer"]) else 0.0
                
                perm_adaptive_totals["sc"] += adaptive_correctness
                perm_adaptive_totals["cost"] += adaptive_ans["tokens_consumed"]
                perm_adaptive_totals["stopped_at"] += adaptive_ans["stopped_at"]
                
                adaptive_stopping_dst.append(adaptive_ans["stopped_at"])
                if sc_correctness == 1.0 and adaptive_correctness == 0.0:
                    incorrect_stopping.append(q_data)
            
        if budget == 32:
            to_test = ["L_i", "K_i", "n_generated_text_tokens"]
            for m in to_test:
                current = selected.sort_values(by=[m], ascending=False)
                top_quartile = current[:8]
                bot_quartile = current[-8:]
                
                top_acc = sum(verifier_fn(gold_ans, ans) for ans in top_quartile["normalized_answer"])
                bot_acc = sum(verifier_fn(gold_ans, ans) for ans in bot_quartile["normalized_answer"])
                
                within_question_diff = top_acc - bot_acc
                
                key = f"assoc-{m}"
                if key not in totals:
                    totals[key] = []
                totals[key].append(within_question_diff)
                
                # length-only control
                if m == "n_generated_text_tokens":
                    shortest_acc = 1.0 if verifier_fn(gold_ans, current.iloc[-1]["normalized_answer"]) else 0.0
                    key = f"assoc-{m}-shortest"
                    if key not in totals:
                        totals[key] = 0.0
                    totals[key] += shortest_acc
            
        for k in perm_totals:
            if k.startswith("assoc-"): continue
            totals[k] += perm_totals[k] / N_perm
        for k in perm_sel_fails:
            sel_fails[k] += perm_sel_fails[k] / N_perm
        for k in perm_sam_fails:
            sam_fails[k] += perm_sam_fails[k] / N_perm
        
        if adaptive_stop_fn is not None:
            for k in perm_adaptive_totals:
                adaptive_totals[k] += perm_adaptive_totals[k] / N_perm
    
    if budget == 32:
        association_metrics = {
            k: v for k, v in paired_confidence_intervals(np.array(totals["assoc-L_i"]), np.array(totals["assoc-K_i"])).items()
        }
        association_metrics = {
            **association_metrics,
            "assoc-L_i": np.mean(totals["assoc-L_i"]),
            "assoc-K_i": np.mean(totals["assoc-K_i"]),
            "assoc-n_generated_text_tokens": np.mean(totals["assoc-n_generated_text_tokens"]),
        }
    
        totals = {
            **totals,
            **association_metrics,
        }
    
    readable_k = {
        "fa": "FirstAns",
        "sc": "SelfConsist",
        "oracle": "Oracle",
        "cost": "Tokens",
        "logp": "Highest-Logp",
        "scert": "Highest-SelfCert",
        "rwc": "Highest-SelfCert-RWC",
        "distinct": "DistinctGroups",
        "assoc-L_i": "Association-Logp",
        "assoc-K_i": "Association-SelfCert",
        "assoc-n_generated_text_tokens": "Association-Length",
        "assoc-n_generated_text_tokens-shortest": "Association-Control-Length",
        "mean": "Association-Avg-Difference",
        "upper": "Association-Upper-CI",
        "lower": "Association-Lower-CI",
        "stopped_at": "Adaptive-Stopping-At",
        "vis": "VisionOnly",
        "visconf": "VisConf",
        "vis_rwc": "VisionOnly-RWC",
        "visconf_rwc": "VisConf-RWC",
    }
    
    n_q = len(unique_sids)
    metrics = {"Budget": budget}
    
    if adaptive_stop_fn is not None:
        metrics["AdaptiveStopping"] = {readable_k[k]: v / n_q for k, v in adaptive_totals.items()}
    
    for k in totals:
        readable = readable_k[k]
        
        if hasattr(totals[k], "tolist"):
            totals[k] = totals[k].tolist()
        metrics[f"Avg-{readable}"] = totals[k] / n_q
        
        if k in sel_fails:
            if hasattr(sel_fails[k], "tolist"):
                sel_fails[k] = sel_fails[k].tolist()
            metrics[f"SelectionFailure-{readable}"] = sel_fails[k] / n_q
        
        if k in sam_fails:
            if hasattr(sam_fails[k], "tolist"):
                sam_fails[k] = sam_fails[k].tolist()
            metrics[f"SamplingFailure-{readable}"] = sam_fails[k] / n_q
    
    return metrics, {"adaptive-dst": adaptive_stopping_dst, "adaptive-incorrect-early": incorrect_stopping}
