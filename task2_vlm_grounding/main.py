import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, random_split
import tqdm

from task2_vlm_grounding.train import calibrate_connector_magnitude, train_model
from task2_vlm_grounding.data.feature import CachedFeaturesDataset
from task2_vlm_grounding.utils import append_jsonl, set_seed
from task2_vlm_grounding.models import load_connector, load_vision_encoder, load_llm, make_lora
from task2_vlm_grounding.data.multitask_sampler import MultiTaskCurriculumSampler
from task2_vlm_grounding.data.caption import collate_captioning, load_coco_captioning
from task2_vlm_grounding.data.grounding import collate_grounding, load_coco_grounding
from task2_vlm_grounding.data.vqa import collate_vqa_pairs, load_vqa
from task2_vlm_grounding.evaluation import evaluate_final


def extract_features(image, processor, encoder):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.float16) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = encoder(**inputs)
        patch_features = outputs.last_hidden_state.detach().cpu().to(torch.float16)

    return patch_features

def cache_features(dataset, encoder, processor, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    encoder.eval()
    
    for idx in tqdm.trange(len(dataset)):
        item = dataset[idx]
        # grounding
        if len(item[0]) == 4 and len(item[1]) == 4:
            for i, cur_item in enumerate(item):
                targ_cache_path = os.path.join(cache_dir, f"targ_{idx}_{i}.json")
                img_cache_path = os.path.join(cache_dir, f"img_{idx}_{i}.pt")
                mismatch_cache_path = os.path.join(cache_dir, f"mismatch_{idx}_{i}.pt")
                if os.path.exists(img_cache_path):
                    continue
                
                image = cur_item[0]
                target = cur_item[1:-1]
                mismatch = cur_item[-1]
                
                patch_features = extract_features(image, processor, encoder)
                mismatch_patch_features = extract_features(mismatch, processor, encoder)

                torch.save(patch_features, img_cache_path)
                torch.save(mismatch_patch_features, mismatch_cache_path)
                with open(targ_cache_path, "w") as f:
                    f.write(json.dumps(target))
        # vqa
        elif isinstance(item[0], dict):
            for i, cur_item in enumerate(item):
                image = cur_item["image"]
                mismatch = cur_item["unrelated_image"]
                target = {k: v for k, v in cur_item.items() if k not in ["image", "unrelated_image"]}
                
                patch_features = extract_features(image, processor, encoder)
                mismatch_patch_features = extract_features(mismatch, processor, encoder)

                targ_cache_path = os.path.join(cache_dir, f"targ_{idx}_{i}.json")
                img_cache_path = os.path.join(cache_dir, f"img_{idx}_{i}.pt")
                mismatch_cache_path = os.path.join(cache_dir, f"mismatch_{idx}_{i}.pt")
                if os.path.exists(img_cache_path):
                    continue
                
                torch.save(patch_features, img_cache_path)
                torch.save(mismatch_patch_features, mismatch_cache_path)
                with open(targ_cache_path, "w") as f:
                    f.write(json.dumps(target))
        # captioning
        else:
            targ_cache_path = os.path.join(cache_dir, f"targ_{idx}.json")
            img_cache_path = os.path.join(cache_dir, f"img_{idx}.pt")
            if os.path.exists(img_cache_path):
                continue
            
            image = item[0]
            target = item[1]
            
            patch_features = extract_features(image, processor, encoder)
                
            torch.save(patch_features, img_cache_path)
            with open(targ_cache_path, "w") as f:
                f.write(json.dumps(target))

    return torch.cuda.max_memory_allocated()

def create_cached(CACHE_ROOT, train_ds, val_ds, collate_fn, stage, dev_frac=None, test_ds=None):
    torch.cuda.reset_peak_memory_stats()
    
    processor, encoder = load_vision_encoder()
    
    peak_vram = torch.cuda.max_memory_allocated()
    start = time.perf_counter()

    if not (CACHE_ROOT / "train").exists():
        print(f"Caching train")
        vram = cache_features(train_ds, encoder, processor, cache_dir=CACHE_ROOT / "train")
        peak_vram = max(peak_vram, vram)
    if not (CACHE_ROOT / "val").exists():
        print(f"Caching dev")
        vram = cache_features(val_ds, encoder, processor, cache_dir=CACHE_ROOT / "val")
        peak_vram = max(peak_vram, vram)
    if test_ds is not None and not (CACHE_ROOT / "test").exists():
        print(f"Caching test")
        vram = cache_features(test_ds, encoder, processor, cache_dir=CACHE_ROOT / "test")
        peak_vram = max(peak_vram, vram)

    del encoder
    torch.cuda.empty_cache()

    num_workers = 8

    val_ds = CachedFeaturesDataset(CACHE_ROOT / "val", stage)
    if test_ds is not None:
        dev_loader = DataLoader(dataset=val_ds, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_ds, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif dev_frac is not None:
        dev_size = int(len(val_ds) * dev_frac)
        test_size = len(val_ds) - dev_size
        print(f"Dev={dev_size}, Test={test_size}")
        dev_ds, test_ds = random_split(val_ds, [dev_size, test_size])
        dev_loader = DataLoader(dataset=dev_ds, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_ds, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif (CACHE_ROOT / "test").exists():
        test_ds = CachedFeaturesDataset(CACHE_ROOT / "test", stage)
        dev_loader = DataLoader(dataset=val_ds, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_ds, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        dev_loader = DataLoader(dataset=val_ds, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = None
    
    train_loader = DataLoader(dataset=CachedFeaturesDataset(CACHE_ROOT / "train", stage), batch_size=8, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    print(f"Train={len(train_loader.dataset)}, Dev={len(dev_loader.dataset)}, Test={len(test_loader.dataset) if test_loader else 'N/A'}")

    duration = time.perf_counter() - start

    return train_loader, dev_loader, test_loader, peak_vram, duration

def run_stage1(connector, llm, tokenizer, train_loader, dev_loaders):
    stage1_updates = 2_250
    train_model(
        connector, llm, tokenizer,
        train_loader, dev_loaders,
        stage=1, total_steps=stage1_updates,
        connector_lr=1e-3, llm_lr=None,
        accum_steps=2, batch_size=8,
        ckpt_dir="./checkpoints/task2/stage-1/",
        resume_training=True,
    )

    return stage1_updates
    
def run_stage2(connector, llm, tokenizer, train_caption_loader, train_presence_loader, train_counting_loader, dev_loaders, stage1_updates):
    train_pools = {
        "presence": list(range(len(train_presence_loader.dataset))),
        "counting": list(range(len(train_counting_loader.dataset))),
        "caption": list(range(len(train_caption_loader.dataset)))
    }
    train_dataset_pools = {
        "presence": train_presence_loader.dataset,
        "counting": train_counting_loader.dataset,
        "caption": train_caption_loader.dataset,
    }
    
    train_loader = MultiTaskCurriculumSampler(train_dataset_pools, train_pools, stage=2, seed=6304)
    
    stage2_updates = 1_500
    train_model(
        connector, llm, tokenizer,
        train_loader, dev_loaders,
        stage=2, total_steps=stage2_updates, prev_stage_updates=stage1_updates,
        connector_lr=1e-4, llm_lr=2e-5,
        accum_steps=2, batch_size=8,
        ckpt_dir="./checkpoints/task2/stage-2/",
        resume_training=True,
    )

    return stage2_updates

def run_stage3(connector, llm, tokenizer, train_caption_loader, train_presence_loader, train_counting_loader, train_vqa_loader, dev_loaders, stage2_updates):
    train_pools = {
        "vqa": list(range(len(train_vqa_loader.dataset))),
        "presence": list(range(len(train_presence_loader.dataset))),
        "counting": list(range(len(train_counting_loader.dataset))),
        "caption": list(range(len(train_caption_loader.dataset))),
    }
    train_dataset_pools = {
        "vqa": train_vqa_loader.dataset,
        "presence": train_presence_loader.dataset,
        "counting": train_counting_loader.dataset,
        "caption": train_caption_loader.dataset,
    }
    
    train_loader = MultiTaskCurriculumSampler(train_dataset_pools, train_pools, stage=3, seed=6304)
    
    stage3_updates = 750
    train_model(
        connector, llm, tokenizer,
        train_loader, dev_loaders,
        stage=3, total_steps=stage3_updates, prev_stage_updates=stage2_updates,
        connector_lr=5e-5, llm_lr=1e-5,
        accum_steps=2, batch_size=8,
        ckpt_dir="./checkpoints/task2/stage-3/",
        resume_training=True,
    )

    return stage3_updates

def load_caption_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT):
    CACHE_ROOT = Path("data/task2/cache/captions")
    
    print("Loading captioning...")
    if not CACHE_ROOT.exists():
        train_captions = load_coco_captioning(
            image_root=TRAIN_ROOT,
            ann_file=COCO_ROOT / "annotations/captions_train2014.json",
            is_train=True,
        )
        dev_captions = load_coco_captioning(
            image_root=VAL_ROOT,
            ann_file=COCO_ROOT / "annotations/captions_val2014.json",
            is_train=False
        )
    else:
        train_captions, dev_captions = None, None  # already cached

    train_loader, dev_loader, _, peak_vram, duration = create_cached(CACHE_ROOT, train_captions, dev_captions, collate_captioning, stage=1, dev_frac=None)

    return train_loader, dev_loader, peak_vram, duration

def load_grounding_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT):
    CACHE_ROOT = Path("data/task2/cache/grounding")
    
    if not CACHE_ROOT.exists():
        print("Loading presence grounding...")
        train_presence = load_coco_grounding(
            image_root=TRAIN_ROOT,
            ann_file=COCO_ROOT / "annotations/instances_train2014.json",
            sampling_pool="presence",
            split="train",
        )
        dev_presence = load_coco_grounding(
            image_root=VAL_ROOT,
            ann_file=COCO_ROOT / "annotations/instances_val2014.json",
            sampling_pool="presence",
            split="dev",
        )
        test_presence = load_coco_grounding(
            image_root=VAL_ROOT,
            ann_file=COCO_ROOT / "annotations/instances_val2014.json",
            sampling_pool="presence",
            split="test",
            exclude_image_ids=dev_presence.all_img_ids,
        )
        
        print("Loading counting grounding...")
        train_counting = load_coco_grounding(
            image_root=TRAIN_ROOT,
            ann_file=COCO_ROOT / "annotations/instances_train2014.json",
            sampling_pool="counting",
            split="train",
        )
        dev_counting = load_coco_grounding(
            image_root=VAL_ROOT,
            ann_file=COCO_ROOT / "annotations/instances_val2014.json",
            sampling_pool="counting",
            split="dev",
        )
        test_counting = load_coco_grounding(
            image_root=VAL_ROOT,
            ann_file=COCO_ROOT / "annotations/instances_val2014.json",
            sampling_pool="counting",
            split="test",
            exclude_image_ids=dev_counting.all_img_ids,
        )
    else:
        train_presence, dev_presence, test_presence, train_counting, dev_counting, test_counting = None, None, None, None, None, None  # already cached

    print("Caching presence grounding...")
    train_presence_loader, dev_presence_loader, test_presence_loader, peak_vram_presence, duration_presence = create_cached(CACHE_ROOT / "presence", train_presence, dev_presence,
                                                                                                                            collate_grounding, stage=2, test_ds=test_presence)
    print("Caching counting grounding...")
    train_counting_loader, dev_counting_loader, test_counting_loader, peak_vram_counting, duration_counting = create_cached(CACHE_ROOT / "counting", train_counting, dev_counting,
                                                                                                                            collate_grounding, stage=2, test_ds=test_counting)

    return (
        train_presence_loader, dev_presence_loader, test_presence_loader,
        train_counting_loader, dev_counting_loader, test_counting_loader,
        peak_vram_presence, duration_presence,
        peak_vram_counting, duration_counting,
    )

def load_vqa_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT):
    CACHE_ROOT = Path("data/task2/cache/vqa")

    train_kwargs = dict(
        annotation_path=COCO_ROOT / "annotations/v2_mscoco_train2014_annotations.json",
        question_path=COCO_ROOT / "annotations/v2_OpenEnded_mscoco_train2014_questions.json",
        complementary_pairs_path=COCO_ROOT / "annotations/v2_mscoco_train2014_complementary_pairs.json",
        image_root=TRAIN_ROOT,
        split="train"
    )
    dev_kwargs = dict(
        annotation_path=COCO_ROOT / "annotations/v2_mscoco_val2014_annotations.json",
        question_path=COCO_ROOT / "annotations/v2_OpenEnded_mscoco_val2014_questions.json",
        complementary_pairs_path=COCO_ROOT / "annotations/v2_mscoco_val2014_complementary_pairs.json",
        image_root=VAL_ROOT,
        split="dev"
    )
    test_kwargs = dict(
        annotation_path=COCO_ROOT / "annotations/v2_mscoco_val2014_annotations.json",
        question_path=COCO_ROOT / "annotations/v2_OpenEnded_mscoco_val2014_questions.json",
        complementary_pairs_path=COCO_ROOT / "annotations/v2_mscoco_val2014_complementary_pairs.json",
        image_root=VAL_ROOT,
        split="test",
    )

    print("Loading vqa...")
    
    train_vqa_yn, dev_vqa_yn, test_vqa_yn = None, None, None
    train_vqa_num, dev_vqa_num, test_vqa_num = None, None, None
    train_vqa_other, dev_vqa_other, test_vqa_other = None, None, None

    if not CACHE_ROOT.exists():
        train_vqa_yn = load_vqa(**train_kwargs, keep_ans_type="yes/no")
        train_vqa_num = load_vqa(**train_kwargs, keep_ans_type="number")
        train_vqa_other = load_vqa(**train_kwargs, keep_ans_type="other")
        
        dev_vqa_yn = load_vqa(**dev_kwargs, keep_ans_type="yes/no")
        dev_vqa_num = load_vqa(**dev_kwargs, keep_ans_type="number")
        dev_vqa_other = load_vqa(**dev_kwargs, keep_ans_type="other")

        # ensure image-disjointness from dev because both use val2014 data
        test_vqa_yn = load_vqa(**test_kwargs, exclude_image_ids=list(dev_vqa_yn.get_image_ids()), keep_ans_type="yes/no")
        test_vqa_num = load_vqa(**test_kwargs, exclude_image_ids=list(dev_vqa_num.get_image_ids()), keep_ans_type="number")
        test_vqa_other = load_vqa(**test_kwargs, exclude_image_ids=list(dev_vqa_other.get_image_ids()), keep_ans_type="other")

    cache_kwargs = dict(
        collate_fn=collate_vqa_pairs,
        stage=3,
    )
    train_vqa_loader_yn, dev_vqa_loader_yn, test_vqa_loader_yn, peak_vram_yn, duration_yn = create_cached(**cache_kwargs, CACHE_ROOT=CACHE_ROOT/"yes-no", train_ds=train_vqa_yn, val_ds=dev_vqa_yn, test_ds=test_vqa_yn)
    train_vqa_loader_num, dev_vqa_loader_num, test_vqa_loader_num, peak_vram_num, duration_num = create_cached(**cache_kwargs, CACHE_ROOT=CACHE_ROOT/"number", train_ds=train_vqa_num, val_ds=dev_vqa_num, test_ds=test_vqa_num)
    train_vqa_loader_other, dev_vqa_loader_other, test_vqa_loader_other, peak_vram_other, duration_other = create_cached(**cache_kwargs, CACHE_ROOT=CACHE_ROOT/"other", train_ds=train_vqa_other, val_ds=dev_vqa_other, test_ds=test_vqa_other)

    return (
        train_vqa_loader_yn, dev_vqa_loader_yn, test_vqa_loader_yn, peak_vram_yn, duration_yn,
        train_vqa_loader_num, dev_vqa_loader_num, test_vqa_loader_num, peak_vram_num, duration_num,
        train_vqa_loader_other, dev_vqa_loader_other, test_vqa_loader_other, peak_vram_other, duration_other
    )

def load_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT):
    tr_caption, dv_caption, peak_vram_caption, duration_caption = load_caption_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT)

    (
        tr_presence, dv_presence, ts_presence,
        tr_counting, dv_counting, ts_counting,
        peak_vram_presence, duration_presence,
        peak_vram_counting, duration_counting,
    ) = load_grounding_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT)

    (
        train_vqa_loader_yn, dev_vqa_loader_yn, test_vqa_loader_yn, peak_vram_yn, duration_yn,
        train_vqa_loader_num, dev_vqa_loader_num, test_vqa_loader_num, peak_vram_num, duration_num,
        train_vqa_loader_other, dev_vqa_loader_other, test_vqa_loader_other, peak_vram_other, duration_other
    ) = load_vqa_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT)

    tr_vqa = DataLoader(
        torch.utils.data.ConcatDataset([train_vqa_loader_yn.dataset, train_vqa_loader_num.dataset, train_vqa_loader_other.dataset]),
        batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_vqa_pairs,
    )
    dv_vqa = DataLoader(
        torch.utils.data.ConcatDataset([dev_vqa_loader_yn.dataset, dev_vqa_loader_num.dataset, dev_vqa_loader_other.dataset]),
        batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_vqa_pairs,
    )
    ts_vqa = {
        "yes/no": test_vqa_loader_yn,
        "number": test_vqa_loader_num,
        "other": test_vqa_loader_other,
    }

    append_jsonl(Path("results") / "task2" / "caching-cost.jsonl", {
        "vram_mb": {
            "caption": peak_vram_caption / (1024 ** 2),
            "presence": peak_vram_presence / (1024 ** 2),
            "counting": peak_vram_counting / (1024 ** 2),
            "vqa": max(peak_vram_yn, peak_vram_num, peak_vram_other) / (1024 ** 2),
        },
        "duration": {
            "caption": duration_caption,
            "presence": duration_presence,
            "counting": duration_counting,
            "vqa": sum([duration_yn, duration_num, duration_other]),
        },
    })

    return (
        tr_caption, dv_caption,
        tr_presence, dv_presence, ts_presence,
        tr_counting, dv_counting, ts_counting,
        tr_vqa, dv_vqa, ts_vqa
    )

def main():
    set_seed(6304)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    COCO_ROOT = Path("data/coco")
    TRAIN_ROOT = COCO_ROOT / "train2014"
    VAL_ROOT = COCO_ROOT / "val2014"
    
    (
        tr_caption, dv_caption,
        tr_presence, dv_presence, ts_presence,
        tr_counting, dv_counting, ts_counting,
        tr_vqa, dv_vqa, ts_vqa
    ) = load_loaders(COCO_ROOT, TRAIN_ROOT, VAL_ROOT)

    dev_loaders = {
        "caption": dv_caption,
        "presence": dv_presence,
        "counting": dv_counting,
        "vqa": dv_vqa,
    }
    test_loaders = {
        "presence": ts_presence,
        "counting": ts_counting,
        "vqa": ts_vqa,
    }

    print("Caption:", len(tr_caption.dataset), len(dv_caption.dataset))
    print("Presence:", len(tr_presence.dataset), len(dv_presence.dataset), len(ts_presence.dataset))
    print("Counting:", len(tr_counting.dataset), len(dv_counting.dataset), len(ts_counting.dataset))
    print("VQA:", len(tr_vqa.dataset), len(dv_vqa.dataset))
    print("VQA (y/n): ", len(ts_vqa["yes/no"].dataset))
    print("VQA (number): ", len(ts_vqa["number"].dataset))
    print("VQA (other): ", len(ts_vqa["other"].dataset))

    connector = load_connector()
    tokenizer, llm = load_llm()
    
    calibrate_connector_magnitude(connector, llm, tokenizer, tr_caption, num_batches=10)

    stage1_updates = run_stage1(connector, llm, tokenizer, tr_caption, dev_loaders)
    torch.save(connector.state_dict(), "./checkpoints/task2/stage-1_connector.pt")
    # stage1_updates = 2_250
    # connector.load_state_dict(torch.load("./checkpoints/task2/stage-1_connector.pt", map_location="cuda", weights_only=False))

    llm = make_lora(llm)
    stage2_updates = run_stage2(connector, llm, tokenizer, tr_caption, tr_presence, tr_counting, dev_loaders, stage1_updates)
    llm.save_pretrained("./checkpoints/task2/stage-2_llm")
    torch.save(connector.state_dict(), "./checkpoints/task2/stage-2_connector.pt")
    
    run_stage3(connector, llm, tokenizer, tr_caption, tr_presence, tr_counting, tr_vqa, dev_loaders, stage1_updates+stage2_updates)
    llm.save_pretrained("./checkpoints/task2/stage-3_llm")
    torch.save(connector.state_dict(), "./checkpoints/task2/stage-3_connector.pt")

    evaluate_final(connector, llm, tokenizer, test_loaders, Path("results") / "task1" / "test" / "test.jsonl")

if __name__ == "__main__":
    main()