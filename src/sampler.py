import random
from torch.utils.data import Sampler

class ClassBalancedBatchSampler(Sampler):
    def __init__(self, class_to_indices, batch_size=256, classes_per_batch=8):
        self.class_to_indices = class_to_indices
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = batch_size // classes_per_batch
        self.class_list = list(class_to_indices.keys())

        # Copy & shuffle all indices per classe
        self.shuffled_class_indices = {k: random.sample(v, len(v)) for k, v in class_to_indices.items()}
        self.num_samples = sum(len(v) for v in class_to_indices.values())

        # Serve per evitare di usare due volte lo stesso indice
        self.used_indices = set()

    def __iter__(self):
        batches = []
        remaining = True

        while remaining:
            selected_classes = random.sample(self.class_list, self.classes_per_batch)
            batch = []

            for cls in selected_classes:
                cls_indices = self.shuffled_class_indices[cls]

                # Se la classe ha meno esempi rimasti di quelli richiesti
                if len(cls_indices) < self.samples_per_class:
                    # Rimuovi quelli già usati e reshuffla
                    all_indices = list(set(self.class_to_indices[cls]) - self.used_indices)
                    if len(all_indices) < self.samples_per_class:
                        # Se ancora insufficienti, prendi tutto ciò che puoi
                        take = all_indices
                    else:
                        take = random.sample(all_indices, self.samples_per_class)
                    self.shuffled_class_indices[cls] = list(set(cls_indices) - set(take))
                else:
                    take = cls_indices[:self.samples_per_class]
                    self.shuffled_class_indices[cls] = cls_indices[self.samples_per_class:]

                batch.extend(take)
                self.used_indices.update(take)

            # Stop if batch is incomplete or all data is used
            if len(batch) == self.batch_size:
                batches.append(batch)

            if len(self.used_indices) >= self.num_samples:
                remaining = False

        return iter(batches)

    def __len__(self):
        return self.num_samples // self.batch_size
