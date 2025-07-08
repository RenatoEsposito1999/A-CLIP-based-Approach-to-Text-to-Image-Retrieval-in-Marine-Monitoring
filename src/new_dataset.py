# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

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

class Custom_dataset_augmented(Dataset):
    """ 
    This class is specific to the Flickr30k dataset downloaded from: https://www.kaggle.com/datasets/eeshawn/flickr30k
    The dataset is composed of images and captions.
    The images are in the flickr30k_images folder.
    The captions are in the captions.txt file.
    """

    def __init__(self, base_path, split='train', model="clip", txt_transform=None):
        # make sur flickr30k_images folder exists in the base_path
        base_path = pathlib.Path(base_path)
        if split == "train":
            img_dir = base_path / "Train"
        elif split == "val":
            img_dir = base_path / "Validation"
        elif split == "test":
            img_dir = base_path / "Validation"
        
        img_dir_coco_flicker = img_dir / "COCO-Flickr30k"
        img_dir_turtle = img_dir / "Turtle"
        img_dir_other_turtle = img_dir / "Turtle_other"
        img_dir_debris = img_dir / "Debris"
        img_dir_dolphin = img_dir / "Dolphin"
        img_dir_sea = img_dir / "Sea"
        
        
        annotations_dir = img_dir / "Annotations"
        annotations_coco_flicker =  annotations_dir / "coco_flicker.csv"
        annotations_turtle = annotations_dir / "turtle.csv"
        annotations_debris = annotations_dir / "debris.csv"
        annotations_sea = annotations_dir / "sea.csv"
        annotations_dolphin = annotations_dir / "dolphin.csv"
        annotations_other_turtle = annotations_dir / "other_turtle.csv"
        #annotations_category = annotations_dir / "category_info.json"
        
        if model == "clip":
            self.generic_transform = T.Compose([
                T.Resize((224, 224)),
            ])
        else:
            self.generic_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        self.txt_transform=txt_transform
        category_info ={
            "turtle": -2,
            "debris": -3,
            "sea": -4,
            "dolphin": -1
        }
        print(category_info)
        # load all captions
        '''
            for each dataset, create e dictionaty, where the key is the path of the image and value is a list of 5 captions associated to that image
        '''
        #START TO INSERT FLICKR30 AND COCO
        self.captions_coco_flickr30, self.imgs_coco_flickr30 = self.read_csv(file_name=annotations_coco_flicker, img_dir=img_dir_coco_flicker, nrows=22000)

        #START TO INSERT TURTLE
        self.captions_turtle, self.imgs_turtle = self.read_csv(file_name=annotations_turtle, img_dir=img_dir_turtle)
        '''
        {key: "path_image", value: [caption, category]}
        '''   
        #START TO INSERT OTHER TURTLE
        captions_other_turtle, imgs_other_turtle = self.read_csv(file_name=annotations_other_turtle, img_dir=img_dir_other_turtle) 
        
        #MERGE TURTLE AND OTHER TURTLE
        self.captions_turtle = self.captions_turtle | captions_other_turtle
        self.imgs_turtle = self.imgs_turtle + imgs_other_turtle
        
        #START TO INSERT DEBRIS          
        self.captions_debris, self.imgs_debris = self.read_csv(file_name=annotations_debris, img_dir=img_dir_debris)
    
        #START TO INSERT SEA           
        self.captions_sea, self.imgs_sea = self.read_csv(file_name=annotations_sea, img_dir=img_dir_sea)
         
        #START TO INSERT DOLPHINE        
        self.captions_dolphin, self.imgs_dolphin = self.read_csv(file_name=annotations_dolphin, img_dir=img_dir_dolphin)
        
        print("turtle: ", len(self.imgs_turtle))
        print("debris: ", len(self.imgs_debris))
        print("sea: ", len(self.imgs_sea))
        print("dolphine: ",len(self.imgs_dolphin))
        print("COCO-flickr30", len(self.imgs_coco_flickr30))
        
        self.imgs = self.imgs_coco_flickr30 + self.imgs_debris + self.imgs_dolphin + self.imgs_sea + self.imgs_turtle 
        random.shuffle(self.imgs)
        self.captions = self.captions_coco_flickr30 | self.captions_turtle | self.captions_debris | self.captions_sea | self.captions_dolphin

    def read_csv(self, file_name, img_dir, nrows=None):
        df = pd.read_csv(file_name)
        captions = defaultdict(list)
        for idx,row in df.iterrows():
            image, comment_number, caption, category = row
            captions[img_dir / image].append(caption)
            captions[img_dir / image].append(category)
        imgs = list(captions.keys())
        random.shuffle(imgs)
        if nrows:
            imgs = imgs[:nrows]
        return captions, imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(img_name).convert('RGB')
        img = self.generic_transform(img)
        captions = self.captions[img_name][0]
        category = self.captions[img_name][1]
        if self.txt_transform:
            captions = [self.txt_transform(caption) for caption in captions]
        return img, captions, category

class Collate_fn_nanoclip:
    """    
        Collate class for the dataloader (to be called in the dataloader)
        This will be called for each batch of data
        It will convert the list of images and captions into a single tensor
        The captions will be tokenized and padded to the max_length 
        The images will be stacked into a single tensor
    """
    def __init__(self, tokenizer, max_length=80, captions_to_use='first'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.captions_to_use = captions_to_use
        
    def __call__(self, batch):
        images, captions, cats = zip(*batch)
        images = torch.stack(images)
        cats = torch.tensor(cats)
        if self.captions_to_use == 'first':
            captions = [caption[0] for caption in captions]
        elif self.captions_to_use == 'random':
            captions = [caption[random.randint(0, 4)] for caption in captions]
        elif self.captions_to_use == 'all':
            pass # use all captions
        else:
            raise ValueError("captions_to_use should be one of 'all', 'first', 'random'")
        
        
        # captions are either a list of strings or a list of list of strings
        captions_ids  = []
        masks = []
        if isinstance(captions[0], list): # list of list of strings               
            # multiple captions
            for caption_list in captions:
                caps = [self.tokenizer(caption, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt") for caption in caption_list]
                captions_ids.append(torch.stack([caption['input_ids'].squeeze(0) for caption in caps]))
                masks.append(torch.stack([caption['attention_mask'].squeeze(0) for caption in caps]))
            captions_ids = torch.stack(captions_ids)
            masks = torch.stack(masks)        
        else:
            # single caption
            captions = self.tokenizer(captions, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
            captions_ids = captions['input_ids'].squeeze(0)
            masks = captions['attention_mask'].squeeze(0)
        
        return images, captions_ids, masks, cats
    
    
class Collate_fn_clip:
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
    
    


