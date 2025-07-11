import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from tqdm import tqdm

class NonRepeatingBalancedSampler(Sampler):
    def __init__(self, dataset, fixed_categories=[-1, -2, -3, -4], samples_per_fixed=64, batch_size=256, drop_last=False):
        self.dataset = dataset
        self.samples_per_fixed = samples_per_fixed
        self.coco_samples = batch_size - (samples_per_fixed * len(fixed_categories))
        self.drop_last = drop_last
        self.fixed_categories = fixed_categories
        self.category_indices = defaultdict(list)
        for idx, img_name in tqdm(enumerate(dataset.imgs), total=len(dataset.imgs)):
            category = dataset.captions[img_name][1]
            self.category_indices[category].append(idx)

        self.coco_categories = [cat for cat in self.category_indices.keys() if cat >= 0]
        self.reset()  # Inizializza gli indici disponibili e usati

    def reset(self):
        """Resetta gli indici usati all'inizio di una nuova epoca."""
        self.available_indices = {}
        self.used_indices = set()  # Tiene traccia di tutti gli indici usati nell'epoca corrente

        for cat in self.category_indices:
            idxs = self.category_indices[cat].copy()
            np.random.shuffle(idxs)
            self.available_indices[cat] = idxs

    def _take_available_samples(self, category, num_samples):
        """Prende `num_samples` dalla categoria, evitando ripetizioni."""
        taken = []
        remaining = num_samples
        '''if(category == -2):
            print("RIMANENTI TURTLE: ", len(self.available_indices[category]))'''
        while remaining > 0 and len(self.available_indices[category]) > 0:
            idx = self.available_indices[category].pop(0)
            if idx not in self.used_indices:
                taken.append(idx)
                self.used_indices.add(idx)
                remaining -= 1

        return taken

    def _sample_from_coco(self, num_samples):
        """Campiona da COCO senza ripetere indici già usati."""
        coco_indices = []
        remaining = 0
        for cat in self.coco_categories:
            coco_indices.extend(self.available_indices[cat])
    
        np.random.shuffle(coco_indices)
        selected = []
        for idx in coco_indices:
            if idx not in self.used_indices:
                remaining += 1
        #print("COCO REMAINING: ", remaining)
        for idx in coco_indices:
            if idx not in self.used_indices:
                selected.append(idx)
                self.used_indices.add(idx)
                if len(selected) >= num_samples:
                    break
        #print(len(selected))
        return selected

    def __iter__(self):
        self.reset()  # Resetta all'inizio di ogni epoca
        batch_count = 0
        flag = False
        while True:
            batch = []

            # 1. Campiona dalle categorie fisse (se esaurite, riempi con COCO)
            for cat in self.fixed_categories:
                taken = self._take_available_samples(cat, self.samples_per_fixed)
                batch.extend(taken)

                # Se non ne ha abbastanza, riempi con COCO
                if len(taken) < self.samples_per_fixed:
                    if cat == -2:
                        flag = True
                    needed = self.samples_per_fixed - len(taken)
                    '''extra_coco = self._sample_from_coco(needed)
                    batch.extend(extra_coco)'''
                    extra_turtle = self._take_available_samples(-2, needed)
                    batch.extend(extra_turtle)
            
             
            # 2. Aggiungi altri sample COCO (opzionale)
            if self.coco_samples > 0:
                extra_coco = self._sample_from_coco(self.coco_samples)
                batch.extend(extra_coco)
            
            if flag:
                break 

            # Se il batch è vuoto, termina
            if len(batch) == 0:
                break
            np.random.shuffle(batch)
            yield batch
            batch_count += 1

    def __len__(self):
        # Stima conservativa del numero di batch
        total_samples = sum(len(idxs) for idxs in self.category_indices.values())
        batch_size = (len(self.fixed_categories) * self.samples_per_fixed) + self.coco_samples
        return total_samples // batch_size