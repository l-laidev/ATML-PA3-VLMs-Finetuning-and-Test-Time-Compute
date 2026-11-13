import json
import os
import torch


class CachedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, stage):
        self.cache_dir = cache_dir
        self.stage = stage
        fnames = [f for f in os.listdir(cache_dir) if f.startswith("img_") and f.endswith(".pt") and not f.endswith("_1.pt")]
        self.num_samples = len(fnames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.stage == 1:
            img_cache_path = os.path.join(self.cache_dir, f"img_{idx}.pt")
            targ_cache_path = os.path.join(self.cache_dir, f"targ_{idx}.json")

            patch_features = torch.load(img_cache_path)
            with open(targ_cache_path, "r") as f:
                target = json.load(f)

            return patch_features, target
        else:
            pairs = [None, None]
            for i in range(2):
                img_cache_path = os.path.join(self.cache_dir, f"img_{idx}_{i}.pt")
                mis_cache_path = os.path.join(self.cache_dir, f"mismatch_{idx}_{i}.pt")
                targ_cache_path = os.path.join(self.cache_dir, f"targ_{idx}_{i}.json")
    
                patch_features = torch.load(img_cache_path)
                mis_patch_features = torch.load(mis_cache_path)
                with open(targ_cache_path, "r") as f:
                    target = json.load(f)
    
                if self.stage == 3:
                    pairs[i] = {
                        **target,
                        "image": patch_features,
                        "mismatch": mis_patch_features,
                    }
                else:
                    pairs[i] = (patch_features, target[0], target[1], mis_patch_features)
            return pairs
