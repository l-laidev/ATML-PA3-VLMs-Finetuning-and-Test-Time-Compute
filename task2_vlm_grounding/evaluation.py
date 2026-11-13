"""
Evaluation and metrics suite for PA3 Task 2: Vision-Language Models and Visual Grounding.
Implements the four evaluation image conditions, VQA soft consensus scoring, 
reference NLL, image-swap penalties, and bootstrap confidence intervals.
"""

import torch
import numpy as np
import tqdm
from typing import List, Dict, Any, Tuple

from pathlib import Path
from task2_vlm_grounding.utils import append_jsonl, normalize_answer, process_batch_generate, process_batch_loss


def vqa_soft_consensus(predicted_answer: str, ref_answers: List[str]) -> float:
    norm_pred = normalize_answer(predicted_answer)
    norm_refs = [normalize_answer(ans) for ans in ref_answers]
    
    total_score = 0.0
    for i in range(len(norm_refs)):
        matches = sum(1 for j, ans in enumerate(norm_refs) if j != i and norm_pred == ans)
        score_i = min(1.0, matches / 3.0)
        total_score += score_i
        
    return total_score / len(norm_refs)

def normalized_exact_match(predicted_answer: str, target_answer: str) -> float:
    return 1.0 if normalize_answer(predicted_answer) == normalize_answer(target_answer) else 0.0

def compute_conf_intervals(
    scores: List[Tuple[int | float, int | float]],
    num_resamples: int = 4000,
) -> Tuple[float, float, float]:
    observed = np.mean(scores)
    rng = np.random.default_rng(seed=6304)
    
    vals = []
    for _ in tqdm.trange(num_resamples, total=num_resamples, desc="Computing CI", leave=False):
        idxs = rng.choice(len(scores), size=len(scores), replace=True)
        pairs = np.array(scores, dtype=np.float32)[idxs]
        vals.append(np.mean(pairs))
        
    ci_lower = np.percentile(vals, 2.5)
    ci_upper = np.percentile(vals, 97.5)
    
    return {
        "mean": observed.tolist(),
        "lower": ci_lower.tolist(),
        "upper": ci_upper.tolist(),
    }


@torch.inference_mode()
def evaluate_caption(connector, llm, tokenizer, caption_loader):
    iterator = iter(caption_loader)
    total_loss = 0.0
    total = 0
    for images, captions in tqdm.tqdm(iterator, total=len(caption_loader), desc="Caption Dev", leave=False):
        total += 1
        targets = list(captions)
        prompts = ["Describe this image." for _ in range(len(targets))]

        visual_embs = connector(images.to(llm.device, dtype=torch.float32)).to(torch.float16)
        batch_loss = process_batch_loss(llm, tokenizer, prompts, targets, visual_embs, is_caption=True)
        total_loss += batch_loss
    return total_loss.item()

@torch.inference_mode()
def evaluate_paired(connector, llm, tokenizer, loader):
    iterator = iter(loader)
    correct_nll, complementary_nll, no_img_nll = 0.0, 0.0, 0.0
    swap_penalty = 0.0
    total = 0
    for (A, B) in tqdm.tqdm(iterator, total=len(loader), desc="Paired Dev", leave=False):
        if isinstance(A[0], dict):  # vqa
            imgsA, promptsA, targsA = (
                torch.stack([a["image"] for a in A]),
                [a["question"] for a in A],
                [a["target"] for a in A],
            )
            imgsB, promptsB, targsB = (
                torch.stack([b["image"] for b in B]),
                [b["question"] for b in B],
                [b["target"] for b in B],
            )
        else:  # grounding
            imgsA, promptsA, targsA = A[0], A[1], A[2]
            imgsB, promptsB, targsB = B[0], B[1], B[2]
        
        visual_embsA = connector(imgsA.to(llm.device, dtype=torch.float32)).to(torch.float16)
        visual_embsB = connector(imgsB.to(llm.device, dtype=torch.float32)).to(torch.float16)

        assert imgsA.shape[0] == imgsB.shape[0]  # same batch size

        correct_nllA = process_batch_loss(llm, tokenizer, promptsA, targsA, visual_embsA)
        correct_nllB = process_batch_loss(llm, tokenizer, promptsB, targsB, visual_embsB)
        
        complementary_nllA = process_batch_loss(llm, tokenizer, promptsA, targsA, visual_embsB)
        complementary_nllB = process_batch_loss(llm, tokenizer, promptsB, targsB, visual_embsA)
        
        no_img_nllA = process_batch_loss(llm, tokenizer, promptsA, targsA, None)
        no_img_nllB = process_batch_loss(llm, tokenizer, promptsB, targsB, None)

        correct_nll += correct_nllA + correct_nllB
        complementary_nll += complementary_nllA + complementary_nllB
        no_img_nll += no_img_nllA + no_img_nllB
        swap_penalty += (complementary_nllA - correct_nllA) + (complementary_nllB - correct_nllB)

        total += (imgsA.shape[0] + imgsB.shape[0])

    correct_nll, complementary_nll, no_img_nll, swap_penalty = correct_nll/total, complementary_nll/total, no_img_nll/total, swap_penalty/total

    return correct_nll.item(), complementary_nll.item(), no_img_nll.item(), swap_penalty.item()


@torch.inference_mode()
def evaluate_dev(connector, llm, tokenizer, dev_loaders, main_run_updates, stage, path):
    connector_was_training = connector.training
    llm_was_training = llm.training

    connector.eval()
    llm.eval()

    caption_loss = evaluate_caption(connector, llm, tokenizer, dev_loaders["caption"])
    
    correct_nll_presence, complementary_nll_presence, no_img_nll_presence, swap_penalty_presence = evaluate_paired(connector, llm, tokenizer, dev_loaders["presence"])
    correct_nll_counting, complementary_nll_counting, no_img_nll_counting, swap_penalty_counting = evaluate_paired(connector, llm, tokenizer, dev_loaders["counting"])
    correct_nll_grounding = 1/2 * (correct_nll_presence + correct_nll_counting)
    complementary_nll_grounding = 1/2 * (complementary_nll_presence + complementary_nll_counting)
    no_img_nll_grounding = 1/2 * (no_img_nll_presence + no_img_nll_counting)
    swap_penalty_grounding = 1/2 * (swap_penalty_presence + swap_penalty_counting)

    correct_nll_vqa, complementary_nll_vqa, no_img_nll_vqa, swap_penalty_vqa = evaluate_paired(connector, llm, tokenizer, dev_loaders["vqa"])

    append_jsonl(path, {
        "main_run_updates": main_run_updates,
        "caption": caption_loss,
        "correct": {
            "presence": correct_nll_presence,
            "counting": correct_nll_counting,
            "grounding": correct_nll_grounding,
            "vqa": correct_nll_vqa,
        },
        "complementary": {
            "presence": complementary_nll_presence,
            "counting": complementary_nll_counting,
            "grounding": complementary_nll_grounding,
            "vqa": complementary_nll_vqa,
        },
        "no_img": {
            "presence": no_img_nll_presence,
            "counting": no_img_nll_counting,
            "grounding": no_img_nll_grounding,
            "vqa": no_img_nll_vqa,
        },
        "swap_penalty": {
            "presence": swap_penalty_presence,
            "counting": swap_penalty_counting,
            "grounding": swap_penalty_grounding,
            "vqa": swap_penalty_vqa,
        },
    })

    if connector_was_training:
        connector.train()
    if llm_was_training:
        llm.train()

@torch.inference_mode()
def evaluate_greedy(connector, llm, tokenizer, loader):
    iterator = iter(loader)
    pairwise_metrics = {
        k: {
            "pair_acc": [],
            "identical_rate": [],
            "empty_rate": [],
            "ref_nll": [],
            "generation_time": [],
        }
        for k in ["correct", "complementary", "no_img", "mismatched"]
    }
    for (A, B) in tqdm.tqdm(iterator, total=len(loader), desc="Test Eval"):
        is_vqa = isinstance(A[0], dict)
        if is_vqa:
            imgsA, promptsA, targsA, mismatchA = (
                torch.stack([a["image"] for a in A]),
                [a["question"] for a in A],
                [a["target"] for a in A],
                torch.stack([a["mismatch"] for a in A]),
            )
            imgsB, promptsB, targsB, mismatchB = (
                torch.stack([b["image"] for b in B]),
                [b["question"] for b in B],
                [b["target"] for b in B],
                torch.stack([b["mismatch"] for b in B]),
            )
        else:
            imgsA, promptsA, targsA, mismatchA = A[0], A[1], A[2], A[3]
            imgsB, promptsB, targsB, mismatchB = B[0], B[1], B[2], A[3]
        
        visual_embsA = connector(imgsA.to(llm.device, dtype=torch.float32)).to(torch.float16)
        visual_embsB = connector(imgsB.to(llm.device, dtype=torch.float32)).to(torch.float16)
        
        mismatch_visual_embsA = connector(mismatchA.to(llm.device, dtype=torch.float32)).to(torch.float16)
        mismatch_visual_embsB = connector(mismatchB.to(llm.device, dtype=torch.float32)).to(torch.float16)

        assert imgsA.shape[0] == imgsB.shape[0]  # same batch size

        for condition, (pA, pB), (vA, vB), (tA, tB) in zip(
            ["correct", "complementary", "no_img", "mismatched"] + ["correct", "complementary", "no_img", "mismatched"],
            zip([promptsA, promptsA, promptsA, promptsA], [promptsB, promptsB, promptsB, promptsB]),
            zip([visual_embsA, visual_embsB, None, mismatch_visual_embsA], [visual_embsB, visual_embsA, None, mismatch_visual_embsB]),
            zip([targsA, targsA, targsA, targsA], [targsB, targsB, targsB, targsB]),
        ):
            gA, durA = process_batch_generate(llm, tokenizer, pA, vA)
            gB, durB = process_batch_generate(llm, tokenizer, pB, vB)

            assert len(gA) == imgsA.shape[0]
            
            for i in range(len(gA)):
                for metric in ["pair_acc", "identical_rate", "empty_rate", "ref_nll", "generation_time"]:
                    metric_val = None

                    if metric == "pair_acc":
                        if is_vqa:
                            metric_val = (vqa_soft_consensus(gA[i], A[i]["answers"]), vqa_soft_consensus(gB[i], B[i]["answers"]))
                        else:
                            metric_val = (normalized_exact_match(gA[i], tA[i]), normalized_exact_match(gB[i], tB[i]))
                    elif metric == "identical_rate":
                        metric_val = normalized_exact_match(gA[i], gB[i])
                    elif metric == "empty_rate":
                        metric_val = (len(gA[i])==0, len(gB[i])==0)
                    elif metric == "ref_nll":
                        metric_val = (process_batch_loss(llm, tokenizer, pA[i:i+1], tA[i:i+1], vA[i:i+1] if vA is not None else vA).item(),
                                      process_batch_loss(llm, tokenizer, pB[i:i+1], tB[i:i+1], vB[i:i+1] if vB is not None else vB).item())
                    elif metric == "generation_time":
                        metric_val = (durA, durB)
                    else:
                        raise ValueError(f"Unexpected metric: {metric}")

                    pairwise_metrics[condition][metric].append(metric_val)

    metrics = {}
    for condition in pairwise_metrics:
        if condition not in metrics:
            metrics[condition] = {}
        
        for metric in pairwise_metrics[condition]:
            if metric not in metrics[condition]:
                metrics[condition][metric] = {}
            
            metrics[condition][metric] = compute_conf_intervals(np.array(pairwise_metrics[condition][metric], dtype=np.float32))

    advantages = {
        "acc": compute_conf_intervals(np.array(pairwise_metrics["correct"]["pair_acc"]).astype(np.float32) - np.array(pairwise_metrics["complementary"]["pair_acc"]).astype(np.float32)),
        "nll": compute_conf_intervals(np.array(pairwise_metrics["correct"]["ref_nll"]).astype(np.float32) - np.array(pairwise_metrics["complementary"]["ref_nll"]).astype(np.float32)),
    }

    return metrics, advantages

@torch.inference_mode()
def evaluate_final(connector, llm, tokenizer, test_loaders, path):
    connector_was_training = connector.training
    llm_was_training = llm.training

    connector.eval()
    llm.eval()

    metrics_presence, adv_presence = evaluate_greedy(connector, llm, tokenizer, test_loaders["presence"])
    metrics_counting, adv_counting = evaluate_greedy(connector, llm, tokenizer, test_loaders["counting"])
    metrics_vqa_yn, adv_vqa_yn = evaluate_greedy(connector, llm, tokenizer, test_loaders["vqa"]["yes/no"])
    metrics_vqa_num, adv_vqa_num = evaluate_greedy(connector, llm, tokenizer, test_loaders["vqa"]["number"])
    metrics_vqa_other, adv_vqa_other = evaluate_greedy(connector, llm, tokenizer, test_loaders["vqa"]["other"])

    append_jsonl(path, {
        "presence": {"metrics": metrics_presence, "advantages": adv_presence},
        "counting": {"metrics": metrics_counting, "advantages": adv_counting},
        "vqa": {
            "yes/no": {"metrics": metrics_vqa_yn, "advantages": adv_vqa_yn},
            "number": {"metrics": metrics_vqa_num, "advantages": adv_vqa_num},
            "other": {"metrics": metrics_vqa_other, "advantages": adv_vqa_other},
        },
    })

    if connector_was_training:
        connector.train()
    if llm_was_training:
        llm.train()
