import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pathlib
from collections import defaultdict
from PIL import Image
import random
import pandas as pd
import json
from torchvision import transforms as T
from utils.seed import seed_everything
import csv
import os

class dataset_test(Dataset):
    def __init__(self, base_path, turtle_transform=T.Compose([T.Resize((224, 224)),]), txt_transform=None, generic_transform=T.Compose([T.Resize((224, 224)),]), is_val = False, seed=12345):
        seed_everything(seed)
        base_path = pathlib.Path(base_path)
        #INITIALIZE VARIABLES FOR PATH TO THE ANNOTATIONS
        annotations_dir = base_path / "annotations"
        annotations_test = annotations_dir / "custom_test.csv"
        
        #INITIALIZE THE TRANSFORM
        self.turtle_transform = turtle_transform
        self.generic_transform = generic_transform
        self.txt_transform = txt_transform
        self.is_val = is_val
    
        category_info ={
            "turtle": -2,
            #"debris": -3,
            #"dolphin": -1,
            #"sea": -4
        }
        #print(category_info)
        unique_category = 0
        
        self.captions = defaultdict(list)
        # Turtle
        self.df = pd.read_csv(annotations_test)
        for idx,row in self.df.iterrows():
            image,caption = row
            self.captions[image].append(caption)
            if "Turtle" in image:
                self.captions[image].append(-2)
            else:
                self.captions[image].append(unique_category)
                unique_category+=1
        
        self.imgs = list(self.captions.keys())
        for img in self.imgs:
            if not os.path.isfile(img):
                print(img)
        
        
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_name = self.imgs[index] #Pick the i image
        img = Image.open(img_name).convert('RGB') #Open the image
        
        #Apply some transformation
        if self.is_val:
            img = self.generic_transform(img)
        else:
            if "cropped" in str(img_name):
                img = self.turtle_transform(img)
            else:
                img = self.generic_transform(img)
            
        '''if self.img_transform:
            img = self.img_transform(img)'''
        captions = self.captions[img_name][0]  #Pick the caption
     
        category = self.captions[img_name][1] #Pick the category
        if self.txt_transform:
            captions = [self.txt_transform(caption) for caption in captions]
        return img, captions, category
   
class Collate_fn:
    """    
        Collate class for the dataloader (to be called in the dataloader)
        This will be called for each batch of data
        It will convert the list of images and captions into a single tensor
        The captions will be tokenized and padded to the max_length 
        The images will be stacked into a single tensor
    """
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch):
        images, captions, cats = zip(*batch)
        cats = torch.tensor(cats)
        images = [image for image in images]
        captions = [caption for caption in captions]
        encoding = self.processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        images = encoding["pixel_values"]
        captions_ids = encoding["input_ids"]
        masks = encoding["attention_mask"]
        return images, captions_ids, masks, cats
    
    


