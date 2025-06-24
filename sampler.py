from torch.utils.data import Sampler
import random
import math
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size,n_classes=16):
        
        n_samples = batch_size//n_classes
        self.labels = labels
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        for label in self.label_to_indices:
            random.shuffle(self.label_to_indices[label])

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = batch_size

        # Calcolo di quanti batch possiamo fare in un'epoca
        self.num_batches = math.floor(len(labels) / self.batch_size)


    def __iter__(self):
        used_label_indices_count = {label: 0 for label in self.label_to_indices}
        classes = list(self.label_to_indices.keys())
        batches = []

        for _ in range(self.num_batches):
            selected_classes = random.sample(classes, self.n_classes)
            batch = []
            for cls in selected_classes:
                start = used_label_indices_count[cls]
                end = start + self.n_samples
                if end > len(self.label_to_indices[cls]):
                    # reshuffle se finiti
                    random.shuffle(self.label_to_indices[cls])
                    start = 0
                    end = self.n_samples
                    used_label_indices_count[cls] = 0
                batch.extend(self.label_to_indices[cls][start:end])
                used_label_indices_count[cls] += self.n_samples
            batches.append(batch)

        random.shuffle(batches)
        return iter([idx for batch in batches for idx in batch])

    def __len__(self):
        return self.num_batches * self.batch_size