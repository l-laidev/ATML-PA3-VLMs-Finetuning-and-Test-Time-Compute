from collections import Counter
import json
import random

from PIL import Image
from torch.utils.data import Dataset
import tqdm

from task2_vlm_grounding.data.caption import get_coco_transform
from task2_vlm_grounding.utils import normalize_answer


class VQADataset(Dataset):
    def __init__(self, image_root, ann_file, q_file, complementary_file, transform=None, split="train", exclude_image_ids=[], keep_ans_type=None):
        self.image_root = image_root
        self.transform = transform
        self.split = split
        
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)['annotations']
        with open(q_file, 'r') as f:
            q_data = json.load(f)['questions']
        with open(complementary_file, 'r') as f:
            c_pairs = json.load(f)

        assert split in ["train", "dev", "test"]
        assert keep_ans_type is not None

        mix_for_split = {
            "train": {"yes/no": 640, "number": 256, "other": 640},
            "dev": {"yes/no": 12, "number": 8, "other": 12},
            "test": {"yes/no": 36, "number": 24, "other": 36},
        }
        total_cap = mix_for_split[split][keep_ans_type]
        self.rng = random.Random(6304)
        
        self.pairs = self.build_pairs(q_data, ann_data, c_pairs, cap=total_cap, ans_type=keep_ans_type, exclude_image_ids=exclude_image_ids)

    def build_pairs(self, q_data, ann_data, c_pairs, cap, ans_type, exclude_image_ids):
        def _process_one(answers):
            answers = [ans["answer"] for ans in answers]
            
            normalized_answers = [normalize_answer(ans) for ans in answers]
            counts = Counter(normalized_answers)
            normalized_chosen, n_votes = counts.most_common(1)[0]
            chosen = [ans for ans in answers if normalize_answer(ans) == normalized_chosen][0]

            return normalized_chosen, n_votes, chosen
        
        id2q = {q['question_id']: q for q in q_data}
        id2a = {ann['question_id']: ann for ann in ann_data}
        pairs = []
        for cur_pair in tqdm.tqdm(c_pairs, desc="Creating VQA dataset"):
            qid1, qid2 = cur_pair

            a1, q1 = id2a[qid1], id2q[qid1]
            a2, q2 = id2a[qid2], id2q[qid2]

            if ans_type is not None and (a1["answer_type"] != ans_type or a2["answer_type"] != ans_type):
                continue

            if a1["image_id"] in exclude_image_ids or a2["image_id"] in exclude_image_ids:
                continue

            normalized_chosen_1, n_votes_1, chosen_1 = _process_one(a1["answers"])
            normalized_chosen_2, n_votes_2, chosen_2 = _process_one(a2["answers"])

            a1_unrelated_img_id = a1["image_id"]
            while a1_unrelated_img_id in [a1["image_id"], a2["image_id"]]:
                idx = self.rng.choice(list(range(len(c_pairs))))
                qid = c_pairs[idx][0]
                a1_unrelated_img_id = id2a[qid]["image_id"]
            
            a2_unrelated_img_id = a2["image_id"]
            while a2_unrelated_img_id in [a1["image_id"], a2["image_id"], a1_unrelated_img_id]:
                idx = self.rng.choice(list(range(len(c_pairs))))
                qid = c_pairs[idx][0]
                a2_unrelated_img_id = id2a[qid]["image_id"]

            if normalized_chosen_1 != normalized_chosen_2 and (n_votes_1 >= 8 and n_votes_2 >= 8):
                pairs.append(
                    (
                        {
                            "image_id": a1["image_id"],
                            "question": q1,
                            "target": chosen_1,
                            "answers": [normalize_answer(ans["answer"]) for ans in a1["answers"]],
                            "ans_type": ans_type,
                            "unrelated_image_id": a1_unrelated_img_id,
                        },
                        {
                            "image_id": a2["image_id"],
                            "question": q2,
                            "target": chosen_2,
                            "answers": [normalize_answer(ans["answer"]) for ans in a2["answers"]],
                            "ans_type": ans_type,
                            "unrelated_image_id": a2_unrelated_img_id,
                        },
                    )
                )

                if cap is not None and len(pairs) >= cap:
                    break
        return pairs

    def get_image_ids(self):
        image_ids = set()
        for (pair1, pair2) in self.pairs:
            image_ids.add(pair1["image_id"])
            image_ids.add(pair2["image_id"])
        return image_ids

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample_a, sample_b = self.pairs[idx]
        coco_split = "train" if self.split == "train" else "val"
        
        img_path_a = self.image_root / f"COCO_{coco_split}2014_{str(sample_a['image_id']).zfill(12)}.jpg"
        img_path_b = self.image_root / f"COCO_{coco_split}2014_{str(sample_b['image_id']).zfill(12)}.jpg"
        
        unrelated_img_path_a = self.image_root / f"COCO_{coco_split}2014_{str(sample_a['unrelated_image_id']).zfill(12)}.jpg"
        unrelated_img_path_b = self.image_root / f"COCO_{coco_split}2014_{str(sample_b['unrelated_image_id']).zfill(12)}.jpg"
        
        image_a = Image.open(img_path_a).convert("RGB")
        image_b = Image.open(img_path_b).convert("RGB")
        
        unrelated_image_a = Image.open(unrelated_img_path_a).convert("RGB")
        unrelated_image_b = Image.open(unrelated_img_path_b).convert("RGB")
        
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            
            unrelated_image_a = self.transform(unrelated_image_a)
            unrelated_image_b = self.transform(unrelated_image_b)
            
        sample_a["image"] = image_a
        sample_b["image"] = image_b
        
        sample_a["unrelated_image"] = unrelated_image_a
        sample_b["unrelated_image"] = unrelated_image_b
        
        return sample_a, sample_b


def collate_vqa_pairs(batch):
    # batch contains tuples of (sample_a, sample_b)
    samples_a = [item[0] for item in batch]
    samples_b = [item[1] for item in batch]
    return samples_a, samples_b


def load_vqa(annotation_path, question_path, complementary_pairs_path, image_root, split, exclude_image_ids=[], keep_ans_type=None):
    transform = get_coco_transform()
    dataset = VQADataset(
        image_root,
        annotation_path,
        question_path,
        complementary_pairs_path,
        transform,
        split,
        exclude_image_ids,
        keep_ans_type,
    )
    return dataset
