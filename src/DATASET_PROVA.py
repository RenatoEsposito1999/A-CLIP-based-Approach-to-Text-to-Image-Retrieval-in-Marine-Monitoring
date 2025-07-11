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

class dataset_SPERANZA(Dataset):
    """ 
    This class is specific to the Flickr30k dataset downloaded from: https://www.kaggle.com/datasets/eeshawn/flickr30k
    The dataset is composed of images and captions.
    The images are in the flickr30k_images folder.
    The captions are in the captions.txt file.
    """

    def __init__(self, base_path, split='train', turtle_transform=None, txt_transform=None, generic_transform=None, is_val = False):
        # make sur flickr30k_images folder exists in the base_path
        base_path = pathlib.Path(base_path)
        img_dir = base_path / "images"
        #img_dir_flicker = img_dir / "flickr30k_images"
        img_dir_COCO = img_dir / "COCO"
        img_dir_turtle = img_dir / "Train_cropped"
        img_dir_other_turtle = img_dir / "other_turtle"
        
        annotations_dir = base_path / "annotations"
        #annotations_flicker =  annotations_dir / "captionsFlicker.txt"
        annotations_COCO = annotations_dir / "COCO_with_category.csv"
        annotations_turtle = annotations_dir / "cropped_turtle.csv"
        annotations_debris = annotations_dir / "cropped_debris.csv"
        #annotations_sea = annotations_dir / "cropped_sea.csv"
        annotations_dolphine = annotations_dir / "cropped_dolphine.csv"
        annotations_other_turtle = annotations_dir / "other_turtle.csv"
        annotations_category = annotations_dir / "category_info.json"
        
        '''if not img_dir.exists():
            raise ValueError(f"Cannot find the flickr30k_images folder in {base_path}. Make sure to download the dataset.")'''
        
        
        self.turtle_transform = turtle_transform
        self.generic_transform = generic_transform
        if self.generic_transform is None:
            self.generic_transform  = T.Compose([
                        T.Resize((224, 224)),
                        #T.ToTensor(),
                        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.txt_transform = txt_transform
        self.is_val = is_val
        
        self.split = split
        category_info ={
            "turtle": -2,
            "debris": -3,
            "dolphin": -1
        }
        #print(category_info)
        unique_category = 0
        # load all captions
        '''
            for each dataset, create e dictionaty, where the key is the path of the image and value is a list of 5 captions associated to that image
        '''
        #START TO INSERT FLICKR30
        self.captions_flickr30 = defaultdict(list)
        
        '''with open(annotations_flicker, 'r') as f:
            for line in f.readlines()[1:]: # ignore the header (first line)
                image, caption_number, caption = line.strip().split(',', 2)
                if caption_number == "0":
                    self.captions_flickr30[img_dir_flicker / image].append(caption)
                    self.captions_flickr30[img_dir_flicker / image].append(unique_category)
                    unique_category += 1'''
                
        
        #START TO INSERT TURTLE (TURTLE AND OTHER TURTLE)    
        #self.captions_turtle = defaultdict( [[], []])
        self.captions_turtle = defaultdict(list)
        '''
        {key: "path_image", value: [caption, category]}
        '''
        '''self.df = pd.read_csv(annotations_turtle)
        for idx,row in self.df.iterrows():
            image, comment_number, caption, category = row
            self.captions_turtle[img_dir_turtle / image].append(caption)
            self.captions_turtle[img_dir_turtle / image].append(-2)'''
        
        self.df = pd.read_csv(annotations_other_turtle)
        for idx,row in self.df.iterrows():
            image, comment_number, caption, category = row
            self.captions_turtle[img_dir_other_turtle / image].append(caption)
            self.captions_turtle[img_dir_other_turtle / image].append(-2)
        
        '''#START TO INSERT DEBRIS          
        self.captions_debris = defaultdict(list)
        self.df = pd.read_csv(annotations_debris)
        for idx,row in self.df.iterrows():
            image, comment_number, caption, category = row
            self.captions_debris[img_dir_turtle / image].append(caption)
            self.captions_debris[img_dir_turtle / image].append(-3)
    
        #START TO INSERT SEA           
        self.captions_sea = defaultdict(list)
        self.df = pd.read_csv(annotations_sea)
        for idx,row in self.df.iterrows():
            image, comment_number, caption, category = row
            self.captions_sea[img_dir_turtle / image].append(caption)
            self.captions_sea[img_dir_turtle / image].append(-4)
            
        #START TO INSERT DOLPHINE        
        self.captions_dolphine = defaultdict(list)
        self.df = pd.read_csv(annotations_dolphine)
        for idx,row in self.df.iterrows():
            image, comment_number, caption, category = row
            self.captions_dolphine[img_dir_turtle / image].append(caption)
            self.captions_dolphine[img_dir_turtle / image].append(-1)     
        '''
        #START TO INSERT COCO
        self.captions_COCO = defaultdict(list)
        self.df = pd.read_csv(annotations_COCO)
        for idx,row in self.df.iterrows():
            image, comment_number, caption, category = row
            self.captions_COCO[img_dir_COCO / image].append(caption)
            self.captions_COCO[img_dir_COCO / image].append(unique_category)
            unique_category += 1
        
                
        # get all image names
        #self.imgs_flickr30 = list(self.captions_flickr30.keys())
        #random.shuffle(self.imgs_flickr30)
        self.imgs_turtle = list(self.captions_turtle.keys())
        random.shuffle(self.imgs_turtle)
        '''self.imgs_debris = list(self.captions_debris.keys())
        random.shuffle(self.imgs_debris)
        self.imgs_sea = list(self.captions_sea.keys())
        random.shuffle(self.imgs_sea)
        self.imgs_dolphine = list(self.captions_dolphine.keys())
        random.shuffle(self.imgs_dolphine)'''
        self.imgs_COCO = list(self.captions_COCO.keys())
        random.shuffle(self.imgs_COCO)
        

        # split the dataset
        if split == 'train':
            #self.imgs_flickr30 = self.imgs_flickr30[ : int(0.8 * len(self.imgs_flickr30))]
            self.imgs_turtle = self.imgs_turtle[ : int(0.8 * len(self.imgs_turtle))]
            '''self.imgs_debris = self.imgs_debris[: int(0.8 * len(self.imgs_debris))]
            self.imgs_sea = self.imgs_sea[: int(0.8 * len(self.imgs_sea))]
            self.imgs_dolphine = self.imgs_dolphine[: int(0.8 * len(self.imgs_dolphine))]'''
            self.imgs_COCO = self.imgs_COCO[ : int(0.5 * len(self.imgs_COCO))] # 20k
            #self.imgs_COCO = self.imgs_COCO[:20000]
        elif split == 'val':
            #self.imgs_flickr30 = self.imgs_flickr30[int(0.8 * len(self.imgs_flickr30)) : int(0.9 * len(self.imgs_flickr30))]
            self.imgs_turtle = self.imgs_turtle[int(0.8 * len(self.imgs_turtle)) : int(0.9 * len(self.imgs_turtle))]
            #self.imgs_turtle = random.sample(self.imgs_turtle, 300)
            '''self.imgs_debris = self.imgs_debris[int(0.8 * len(self.imgs_debris)) : int(0.9 * len(self.imgs_debris))]
            self.imgs_sea = self.imgs_sea[int(0.8 * len(self.imgs_sea)) : int(0.9 * len(self.imgs_sea))]
            self.imgs_dolphine = self.imgs_dolphine[int(0.8 * len(self.imgs_dolphine)) : int(0.9 * len(self.imgs_dolphine))]'''
            self.imgs_COCO = self.imgs_COCO[int(0.5 * len(self.imgs_COCO)) : int(0.65 * len(self.imgs_COCO))] # 16k

        elif split == "test":
            #self.imgs_flickr30 = self.imgs_flickr30[int(0.9 * len(self.imgs_flickr30)) : ]
            self.imgs_turtle = self.imgs_turtle[int(0.9 * len(self.imgs_turtle)) : ]
            '''self.imgs_debris = self.imgs_debris[int(0.9 * len(self.imgs_debris)) : ]
            self.imgs_sea = self.imgs_sea[int(0.9 * len(self.imgs_sea)) : ]
            self.imgs_dolphine = self.imgs_dolphine[int(0.9 * len(self.imgs_dolphine)) : ]'''
            self.imgs_COCO = self.imgs_COCO[int(0.9 * len(self.imgs_COCO)) : ]
        else: # use all images
            pass
        
        #SE VUOI FARE QUALCHE TEST CON POCHE IMMAGINI DECOMMENTA QUELLO CHE C'è SOTTO
        # split the dataset
        '''if split == 'train':
            #self.imgs_flickr30 = self.imgs_flickr30[: int(0.1 * len(self.imgs_flickr30))]
            self.imgs_turtle = self.imgs_turtle[: int(0.01 * len(self.imgs_turtle))]
            self.imgs_debris = self.imgs_debris[: int(0.01 * len(self.imgs_debris))]
            self.imgs_sea = self.imgs_sea[: int(0.01 * len(self.imgs_sea))]
            self.imgs_dolphine = self.imgs_dolphine[: int(0.01 * len(self.imgs_dolphine))]
            self.imgs_COCO = self.imgs_COCO[: int(0.01 * len(self.imgs_COCO))]
        elif split == 'val':
            self.imgs_turtle = self.imgs_turtle[int(0.1 * len(self.imgs_turtle)) : int(0.9 * len(self.imgs_turtle))]
            #self.imgs_turtle = random.sample(self.imgs_turtle, 300)
            self.imgs_debris = self.imgs_debris[int(0.8 * len(self.imgs_debris)) : int(0.9 * len(self.imgs_debris))]
            self.imgs_sea = self.imgs_sea[int(0.8 * len(self.imgs_sea)) : int(0.9 * len(self.imgs_sea))]
            self.imgs_dolphine = self.imgs_dolphine[int(0.8 * len(self.imgs_dolphine)) : int(0.9 * len(self.imgs_dolphine))]
            self.imgs_COCO = self.imgs_COCO[int(0.8 * len(self.imgs_COCO)) : int(0.9 * len(self.imgs_COCO))]
        elif split == "test":
            # mantieni il test originale o riduci anche qui se vuoi
            #self.imgs_flickr30 = self.imgs_flickr30[int(0.9 * len(self.imgs_flickr30)) : ]
            self.imgs_turtle = self.imgs_turtle[int(0.9 * len(self.imgs_turtle)) : ]
            self.imgs_debris = self.imgs_debris[int(0.9 * len(self.imgs_debris)) : ]
            self.imgs_sea = self.imgs_sea[int(0.9 * len(self.imgs_sea)) : ]
            self.imgs_dolphine = self.imgs_dolphine[int(0.9 * len(self.imgs_dolphine)) : ]
            self.imgs_COCO = self.imgs_COCO[int(0.9 * len(self.imgs_COCO)) : ]
        else: # use all images
            pass'''
        
        #print("flickr30: ", len(self.imgs_flickr30))
        print("turtle: ", len(self.imgs_turtle))
        '''print("debris: ", len(self.imgs_debris))
        print("sea: ", len(self.imgs_sea))
        print("dolphine: ",len(self.imgs_dolphine))'''
        print("coco", len(self.imgs_COCO))
        #print("flickr30", len(self.imgs_flickr30))
        
        #self.imgs = self.imgs_COCO + self.imgs_turtle + self.imgs_debris + self.imgs_dolphine 
        self.imgs = self.imgs_COCO + self.imgs_turtle 
        #self.imgs = self.imgs_turtle + self.imgs_sea + self.imgs_debris + self.imgs_dolphine + self.imgs_COCO
        random.shuffle(self.imgs)
        #self.captions = self.captions_flickr30 | self.captions_COCO | self.captions_turtle | self.captions_debris | self.captions_sea | self.captions_dolphine
        self.captions = self.captions_COCO | self.captions_turtle
        #self.captions = self.captions_COCO | self.captions_turtle | self.captions_debris | self.captions_sea | self.captions_dolphine
        

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(img_name).convert('RGB')
        if self.is_val:
            img = self.generic_transform(img)
        else:
            if "cropped" in str(img_name):
                img = self.turtle_transform(img)
            else:
                img = self.generic_transform(img)
            
        '''if self.img_transform:
            img = self.img_transform(img)'''
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
    
    


