import random
from torch.utils.data import Sampler

class ClassBalancedBatchSampler(Sampler):
    def __init__(self, class_to_indices, batch_size=256, classes_per_batch=16):
        assert batch_size % classes_per_batch == 0, "batch_size deve essere divisibile per classes_per_batch"
        
        self.class_to_indices = class_to_indices
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = batch_size // classes_per_batch
        self.class_list = list(class_to_indices.keys())
        # Calcola il numero totale di campioni unici
        self.total_unique_samples = sum(len(indices) for indices in class_to_indices.values())

    def __iter__(self):
        shuffled_indices = {cls: random.sample(indices, len(indices)) for cls, indices in self.class_to_indices.items()}
        used_ptr = {cls: 0 for cls in self.class_list}
        total_used_unique = 0

        while total_used_unique < self.total_unique_samples:
            batch = []
            selected_classes = random.sample(self.class_list, self.classes_per_batch)
            class_distribution = {cls: 0 for cls in selected_classes}

            for cls in selected_classes:
                # Prendi campioni freschi se disponibili
                if used_ptr[cls] < len(shuffled_indices[cls]):
                    take = shuffled_indices[cls][used_ptr[cls] : used_ptr[cls] + self.samples_per_class]
                    actual_take = len(take)
                    batch.extend(take)
                    used_ptr[cls] += actual_take
                    total_used_unique += actual_take
                    class_distribution[cls] += actual_take

                    # Se mancano campioni, riempie con reshuffle
                    if actual_take < self.samples_per_class:
                        needed = self.samples_per_class - actual_take
                        random.shuffle(shuffled_indices[cls])
                        batch.extend(shuffled_indices[cls][:needed])
                        used_ptr[cls] = needed
                        class_distribution[cls] += needed
                else:
                    # Solo reshuffle se nessun campione fresco rimasto
                    random.shuffle(shuffled_indices[cls])
                    batch.extend(shuffled_indices[cls][:self.samples_per_class])
                    used_ptr[cls] = self.samples_per_class
                    class_distribution[cls] += self.samples_per_class

            '''# DEBUG: Stampa la distribuzione (FORZATA con flush)
            print("\n=== DEBUG BATCH ===", flush=True)
            print(f"Classi selezionate: {selected_classes}", flush=True)
            for cls, count in class_distribution.items():
                print(f"Classe {cls}: {count} campioni", flush=True)
            print("------------------", flush=True)'''

            yield batch

    def __len__(self):
        # Stima il numero di batch che verranno prodotti
        return (self.total_unique_samples + self.batch_size - 1) // self.batch_size

'''import random
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
        used_indices = set()
        self.shuffled_class_indices = {k: random.sample(v, len(v)) for k, v in self.class_to_indices.items()}
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
                    all_indices = list(set(self.class_to_indices[cls]) - used_indices)
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
                used_indices.update(take)

            # Stop if batch is incomplete or all data is used
            if len(batch) == self.batch_size:
                batches.append(batch)

            if len(used_indices) >= self.num_samples:
                remaining = False
        return iter(batches)

    def __len__(self):
        return self.num_samples // self.batch_size'''

# Esempio di dataset fittizio
'''class_to_indices = {
    0: list(range(3792)),
    1: list(range(5516)),
    2: list(range(8529)),
    3: list(range(4678)),
    4: list(range(2220)),
    5: list(range(1697)),
    6: list(range(2479)),
    7: list(range(1106)),
    8: list(range(2537)),
    9: list(range(3380)),
    10: list(range(2073)),
    11: list(range(2089)),
    12: list(range(27631)),
    13: list(range(627)),
    14: list(range(5106)),
    15: list(range(531))
}
# Istanzia il sampler
sampler = ClassBalancedBatchSampler(class_to_indices)

# Simula un'iterazione del DataLoader
for i, batch in enumerate(sampler):
    print(batch)
    print(f"\nBatch {i}: Primi 5 elementi = {batch[:5]}")
    if i == 1:  # Interrompi dopo 2 batch per il test
        break'''