from collections import Counter
import json
import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import tqdm
from task2_vlm_grounding.data.caption import get_coco_transform


class GroundingDataset(Dataset):
    def __init__(self, image_root, ann_file, sampling_pool="presence", transform=None, split="train", exclude_image_ids=[]):
        self.image_root = image_root
        self.transform = transform
        self.sampling_pool = sampling_pool

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
            
        self.img2file = {img['id']: img['file_name'] for img in coco_data['images']}
        self.cat2name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        self.img2cat = {}
        self.cat2img = {cat: set() for cat in self.cat2name.values()}
        self.img2count = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            cat_name = self.cat2name[cat_id]
            
            if img_id not in self.img2cat:
                self.img2cat[img_id] = set()
                self.img2count[img_id] = Counter()
                
            self.img2cat[img_id].add(cat_name)
            self.img2count[img_id][cat_name] += 1
            self.cat2img[cat_name].add(img_id)

        self.all_img_ids = list(self.img2file.keys())
        self.all_img_ids = [img_id for img_id in self.all_img_ids if img_id not in exclude_image_ids]
        
        self.rng = random.Random(6304)
        cap = None if split == "train" else 24 if split == "dev" else 64
        if cap is not None:
            self.all_img_ids = self.rng.sample(self.all_img_ids, k=cap)
        
        self.presence_samples, self.counting_samples = self._build_pool()
        
        if split != "train":
            self.presence_samples = self.rng.sample(self.presence_samples, k=cap//2)
            self.counting_samples = self.rng.sample(self.counting_samples, k=cap//2)
        else:
            # total training updates will be (opt updates * batch * accum) = 24000
            # sample in advance instead of caching all samples
            cap = 24_000
            self.presence_samples = self.rng.choices(self.presence_samples, k=cap//2)
            self.counting_samples = self.rng.choices(self.counting_samples, k=cap//2)

    def _build_pool(self):
        presence_samples = []
        counting_samples = []
        for img_id in tqdm.tqdm(self.all_img_ids, desc="Building grounding samples"):
            if img_id not in self.img2cat:
                continue
            present_cats = self.img2cat[img_id]
            counts = self.img2count[img_id]
            
            for cat in present_cats:
                # pair with an img without this category
                negative_id = img_id
                while negative_id not in self.img2cat or cat in self.img2cat[negative_id]:
                    negative_id = self.rng.choice(self.all_img_ids)

                unrelated_id = img_id
                while unrelated_id not in self.img2cat or unrelated_id in [img_id, negative_id]:
                    unrelated_id = self.rng.choice(self.all_img_ids)

                negative_unrelated_id = img_id
                while negative_unrelated_id not in self.img2cat or negative_unrelated_id in [img_id, negative_id, unrelated_id]:
                    negative_unrelated_id = self.rng.choice(self.all_img_ids)
                
                presence_samples.append(
                    (
                        {
                            "image_id": img_id,
                            "question": f"Does this image contain {cat}?",
                            "answer": "yes",
                            "mismatch": unrelated_id,
                        },
                        {
                            "image_id": negative_id,
                            "question": f"Does this image contain {cat}?",
                            "answer": "no",
                            "mismatch": negative_unrelated_id,
                        }
                    )
                )

            for cat, count in counts.items():
                if 1 <= count <= 4:
                    # pair with an img with a different count of the same category
                    paired_id = self.rng.choice(list(self.cat2img[cat]))
                    paired_count = self.img2count[paired_id][cat]
                    while paired_id == img_id or (paired_count < 1 or paired_count > 4) or paired_count == count:
                        paired_id = self.rng.choice(list(self.cat2img[cat]))
                        paired_count = self.img2count[paired_id][cat]

                    unrelated_id = img_id
                    while unrelated_id not in self.img2cat or unrelated_id in [img_id, paired_id]:
                        unrelated_id = self.rng.choice(self.all_img_ids)
    
                    negative_unrelated_id = img_id
                    while negative_unrelated_id not in self.img2cat or negative_unrelated_id in [img_id, paired_id, unrelated_id]:
                        negative_unrelated_id = self.rng.choice(self.all_img_ids)
                    
                    counting_samples.append(
                        (
                            {
                                "image_id": img_id,
                                "question": f"How many {cat}s can you see?",
                                "answer": str(count),
                                "mismatch": unrelated_id,
                            },
                            {
                                "image_id": paired_id,
                                "question": f"How many {cat}s can you see?",
                                "answer": str(paired_count),
                                "mismatch": negative_unrelated_id,
                            }
                        )
                    )

        return presence_samples, counting_samples

    def __len__(self):
        if self.sampling_pool == "presence":
            return len(self.presence_samples)
        elif self.sampling_pool == "counting":
            return len(self.counting_samples)
        else:
            raise ValueError(f"Invalid sampling pool: {self.sampling_pool}")

    def __getitem__(self, idx):
        if self.sampling_pool == "presence":
            src = self.presence_samples
        elif self.sampling_pool == "counting":
            src = self.counting_samples
        else:
            raise ValueError(f"Invalid sampling pool: {self.sampling_pool}")

        ret = [None, None]
        for i, pair in enumerate(src[idx]):
            image = load_img(self.img2file[pair["image_id"]], self.image_root, self.transform)
            mismatch_image = load_img(self.img2file[pair["mismatch"]], self.image_root, self.transform)
            ret[i] = (image, pair["question"], pair["answer"], mismatch_image)
        
        return ret

def load_img(fname, root, tform):
    path = os.path.join(root, fname)
    
    image = Image.open(path).convert("RGB")
    if tform:
        image = tform(image)

    return image

def collate_grounding(batch):
    samples = [None, None]
    for pair_idx in range(2):
        images, questions, answers, mismatches = zip(*[item[pair_idx] for item in batch])
        images = torch.stack(images, 0)
        mismatches = torch.stack(mismatches, 0)
        samples[pair_idx] = (images, list(questions), list(answers), mismatches)
    return samples

def load_coco_grounding(image_root, ann_file, sampling_pool, split, exclude_image_ids=[]):
    transform = get_coco_transform()
    dataset = GroundingDataset(image_root, ann_file, transform=transform, sampling_pool=sampling_pool, split=split, exclude_image_ids=exclude_image_ids)
    return dataset
