import random
from typing import List, Any, Dict, Tuple

from task2_vlm_grounding.data.feature import CachedFeaturesDataset


class MultiTaskCurriculumSampler:
    def __init__(self, dataset_pools: Dict[str, CachedFeaturesDataset], pools: Dict[str, List[int]], stage: int, seed: int = 6304):
        self.dataset_pools = dataset_pools
        self.pools = pools
        self.stage = stage
        self.rng = random.Random(seed)
        
        if stage == 2:
            self.weights = {"presence": 0.50, "counting": 0.25, "caption": 0.25}
        elif stage == 3:
            self.weights = {"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}
        else:
            raise ValueError(f"Invalid stage {stage} for multi-task sampler.")

        self.tasks = list(self.weights.keys())
        self.probabilities = list(self.weights.values())

    def sample(self) -> Tuple[str, Any | List[Any]]:
        chosen_task = self.rng.choices(self.tasks, weights=self.probabilities, k=1)[0]

        if chosen_task != "caption":
            pool_item = self.rng.choice(self.pools[chosen_task])
            return chosen_task, self.dataset_pools[chosen_task][pool_item]
        else:
            pool_item_1 = self.rng.choice(self.pools[chosen_task])
            pool_item_2 = self.rng.choice(self.pools[chosen_task])
            while pool_item_2 == pool_item_1:
                pool_item_2 = self.rng.choice(self.pools[chosen_task])
            return chosen_task, (self.dataset_pools[chosen_task][pool_item_1], self.dataset_pools[chosen_task][pool_item_2])
