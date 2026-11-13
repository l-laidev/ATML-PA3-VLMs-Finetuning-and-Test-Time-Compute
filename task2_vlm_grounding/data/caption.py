import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms

import tqdm


class CaptionsDataset(Dataset):
    def __init__(self, image_root, ann_file, transform=None, is_train=True):
        self.coco = dset.CocoCaptions(root=image_root, annFile=ann_file, transform=None)
        self.transform = transform
        self.is_train = is_train
        
        self.samples = []
        total_images = 6_000 if self.is_train else 96
        for idx in tqdm.trange(total_images, desc="Loading caption dataset"):
            _, captions = self.coco[idx]
            if self.is_train:
                for cap_idx in range(3):
                    self.samples.append((idx, captions[cap_idx]))
            else:
                self.samples.append((idx, captions[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, caption = self.samples[idx]
        img, _ = self.coco[img_idx]
        if self.transform:
            img = self.transform(img)
        return img, caption


def get_coco_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # SigLIP normalization
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
    ])

def collate_captioning(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(captions)

def load_coco_captioning(image_root, ann_file, is_train):
    transform = get_coco_transform()
    dataset = CaptionsDataset(image_root=image_root, ann_file=ann_file, transform=transform, is_train=is_train)
    return dataset
