import torch
from torch.utils.data import Sampler
import random
from collections import defaultdict

class BalancedValidationSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.category_indices = defaultdict(list)

        # Organizza gli indici per categoria
        for idx in range(len(dataset)):
            _, _, flag = dataset[idx]
            self.category_indices[flag].append(idx)

        # Controlli
        self.num_batches = len(self.category_indices[-1])  # basato su turtle

        # Shuffle iniziale
        for k in self.category_indices:
            random.shuffle(self.category_indices[k])

        # Indici iniziali per ogni categoria
        self.category_ptr = {k: 0 for k in self.category_indices}

    def __iter__(self):
        batches = []

        for _ in range(self.num_batches):
            batch = []

            # Preleva uno da ciascuna delle classi speciali
            for cat in [-1, -2, -3, -4]:
                indices = self.category_indices[cat]
                ptr = self.category_ptr[cat]
                if ptr >= len(indices):
                    # Se finiti, pesca random tra tutti
                    idx = random.choice(indices)
                else:
                    idx = indices[ptr]
                    self.category_ptr[cat] += 1
                batch.append(idx)

            # Completa con campioni dalla categoria 0
            needed = self.batch_size - 4
            cat0_indices = self.category_indices[0]
            cat0_ptr = self.category_ptr[0]

            if cat0_ptr + needed <= len(cat0_indices):
                batch.extend(cat0_indices[cat0_ptr:cat0_ptr + needed])
                self.category_ptr[0] += needed
            else:
                # Se finiti, rimpiazza casualmente
                remaining = len(cat0_indices) - cat0_ptr
                batch.extend(cat0_indices[cat0_ptr:])
                self.category_ptr[0] = len(cat0_indices)

                still_needed = needed - remaining
                batch.extend(random.choices(cat0_indices, k=still_needed))

            random.shuffle(batch)  # Shuffle dentro il batch
            batches.append(batch)

        return iter(batches)

    def __len__(self):
        return self.num_batches
