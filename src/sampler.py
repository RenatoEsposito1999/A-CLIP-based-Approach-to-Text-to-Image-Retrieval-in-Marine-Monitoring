import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from tqdm import tqdm
'''
    This Sampler creates balanced batch for computing correctly the UniLoss, and guarantee to have same number
    of samples of the same category inside a batch, without repetation.
    For example:
    fixed_categories=[-2]
    samples_per_fixed = 64
    batch_size = 256
    So each batch is composed by 64 samples of category -2 and the remaining by coco (in this example 256 - 64= 192)
'''
class NonRepeatingBalancedSampler(Sampler):
    def __init__(self, dataset, fixed_categories=[-1, -2, -3, -4], samples_per_fixed=64, batch_size=256, drop_last=False):
        self.dataset = dataset
        self.samples_per_fixed = samples_per_fixed #How much sample you want for the categories specified into fixed_categories
        self.coco_samples = batch_size - (samples_per_fixed * len(fixed_categories)) #How much coco samples need to create a batch, coco is used as distractors in order to simulate batch as negative
        self.drop_last = drop_last
        self.fixed_categories = fixed_categories
        #For each category, create a list with di indices of samples inside the dataset
        self.category_indices = defaultdict(list)
        for idx, img_name in tqdm(enumerate(dataset.imgs), total=len(dataset.imgs)):
            category = dataset.captions[img_name][1]
            self.category_indices[category].append(idx)
        
        #create list of categories assigned to coco (each caption-image of COCO is a unique category)
        self.coco_categories = [cat for cat in self.category_indices.keys() if cat >= 0]
        self.reset()  # Initialize the indices used and remaining

    def reset(self):
        """
            Initialize each step the indices of categories used and remaining
        """
        self.available_indices = {} # Initialize dictonary of indices available for creating a batch
        self.used_indices = set()  #This is a set the help to remember which indices are already used

        #For each category initialize the respective list of indices of that category inside the dict self.available_indices
        for cat in self.category_indices:
            idxs = self.category_indices[cat].copy()
            np.random.shuffle(idxs)
            self.available_indices[cat] = idxs

    
    def _take_available_samples(self, category, num_samples):
        """
            This function return the number of samples of that category for constructing a batch
        """
        taken = []
        remaining = num_samples #Num_samples is the number of samples needed for that batch of a specific category
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
        """
            Extract from COCO dataset a number of samples (num_samples) needed to complete the dataset
        """
        coco_indices = []
        remaining = 0
        #Construct the list coco_indices with all indices available of COCO
        for cat in self.coco_categories:
            coco_indices.extend(self.available_indices[cat])
    
        np.random.shuffle(coco_indices)
        selected = []
        
        #This for is only for debug to visualize how much coco is remaining
        for idx in coco_indices:
            if idx not in self.used_indices:
                remaining += 1
        #print("COCO REMAINING: ", remaining)
        
        #Create list selected, that contains the indices of samples COCO needed to complete the batch
        for idx in coco_indices:
            if idx not in self.used_indices:
                selected.append(idx)
                self.used_indices.add(idx)
                if len(selected) >= num_samples:
                    break
        #print(len(selected))
        return selected

    def __iter__(self):
        """
            this method create all batches, the batches are created untill the category turtle is not empty
        """
        self.reset()  # Initialize all the dictionaries
        batch_count = 0
        flag = False
        while True:
            batch = []

            # 1. Sampled from fixed categories (if exhausted, fill with COCO)
            for cat in self.fixed_categories:
                taken = self._take_available_samples(cat, self.samples_per_fixed)
                batch.extend(taken)

                # If insufficient, fill with remaining cageroy -2 (turtle)
                if len(taken) < self.samples_per_fixed:
                    if cat == -2:
                        flag = True
                    needed = self.samples_per_fixed - len(taken)
                    '''extra_coco = self._sample_from_coco(needed)
                    batch.extend(extra_coco)'''
                    extra_turtle = self._take_available_samples(-2, needed)
                    batch.extend(extra_turtle)
            
             
            # 2. Add coco samples
            if self.coco_samples > 0:
                extra_coco = self._sample_from_coco(self.coco_samples)
                batch.extend(extra_coco)
            
            # If turtle is finished stop to create
            if flag:
                break 

            # If can not create other batch stop
            if len(batch) == 0:
                break
            np.random.shuffle(batch)
            yield batch
            batch_count += 1

    def __len__(self):
        # A sort of estimation of the number of batches
        total_samples = sum(len(idxs) for idxs in self.category_indices.values())
        batch_size = (len(self.fixed_categories) * self.samples_per_fixed) + self.coco_samples
        return total_samples // batch_size