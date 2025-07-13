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
    def __init__(self, base_path, split='train', turtle_transform=T.Compose([T.Resize((224, 224)),]), txt_transform=None, generic_transform=T.Compose([T.Resize((224, 224)),]), is_val = False):
        base_path = pathlib.Path(base_path)
        #INITIALIZE VARIABLES FOR PATH TO THE IMAGES
        img_dir_COCO = base_path / "COCO"
        img_dir_turtle = base_path / "Turtle"
        img_dir_other_turtle = base_path / "Turtle_other"
        img_dir_debris = base_path / "Debris"
        img_dir_dolphin = base_path / "Dolphin"
        img_dir_sea = base_path / "Sea"


        #INITIALIZE VARIABLES FOR PATH TO THE ANNOTATIONS
        annotations_dir = base_path / "annotations"
        annotations_COCO = annotations_dir / "COCO.csv"
        annotations_turtle = annotations_dir / "Turtle.csv"
        annotations_debris = annotations_dir / "Debris.csv"
        annotations_sea = annotations_dir / "Sea.csv"
        annotations_dolphine = annotations_dir / "Dolphin.csv"
        annotations_other_turtle = annotations_dir / "Other_turtle.csv"
        
        #INITIALIZE THE TRANSFORM
        self.turtle_transform = turtle_transform
        self.generic_transform = generic_transform
        self.txt_transform = txt_transform
        self.is_val = is_val
        
        self.split = split
        
        category_info ={
            "turtle": -2,
            #"debris": -3,
            #"dolphin": -1,
            #"sea": -4
        }
        #print(category_info)
        unique_category = 0
        
        #for each list images dataset, create e dictionary, where the key is the path of the image and value is the aption associated to that image
        #{key: "path_image", value: [caption, category]}
        self.captions_turtle = defaultdict(list)
        # Turtle
        self.df = pd.read_csv(annotations_turtle)
        for idx,row in self.df.iterrows():
            image,caption = row
            self.captions_turtle[img_dir_turtle / image].append(caption)
            self.captions_turtle[img_dir_turtle / image].append(-2)
        # Other turtle
        self.df = pd.read_csv(annotations_other_turtle)
        for idx,row in self.df.iterrows():
            image, caption = row
            self.captions_turtle[img_dir_other_turtle / image].append(caption)
            self.captions_turtle[img_dir_other_turtle / image].append(-2)
        # Debris          
        self.captions_debris = defaultdict(list)
        self.df = pd.read_csv(annotations_debris)
        for idx,row in self.df.iterrows():
            image, caption = row
            self.captions_debris[img_dir_debris / image].append(caption)
            self.captions_debris[img_dir_debris / image].append(unique_category)
            unique_category += 1
        # Sea           
        self.captions_sea = defaultdict(list)
        self.df = pd.read_csv(annotations_sea)
        for idx,row in self.df.iterrows():
            image, caption = row
            self.captions_sea[img_dir_sea / image].append(caption)
            self.captions_sea[img_dir_sea / image].append(unique_category)
            unique_category += 1
        # Dolphin    
        self.captions_dolphine = defaultdict(list)
        self.df = pd.read_csv(annotations_dolphine)
        for idx,row in self.df.iterrows():
            image, caption = row
            self.captions_dolphine[img_dir_dolphin / image].append(caption)
            self.captions_dolphine[img_dir_dolphin / image].append(unique_category)     
            unique_category += 1
        # COCO
        self.captions_COCO = defaultdict(list)
        self.df = pd.read_csv(annotations_COCO)
        for idx,row in self.df.iterrows():
            image, caption, _= row
            self.captions_COCO[img_dir_COCO / image].append(caption)
            self.captions_COCO[img_dir_COCO / image].append(unique_category)
            unique_category += 1
        
                
        # get all image names
        self.imgs_turtle = list(self.captions_turtle.keys())
        random.shuffle(self.imgs_turtle)
        self.imgs_debris = list(self.captions_debris.keys())
        random.shuffle(self.imgs_debris)
        self.imgs_sea = list(self.captions_sea.keys())
        random.shuffle(self.imgs_sea)
        self.imgs_dolphine = list(self.captions_dolphine.keys())
        random.shuffle(self.imgs_dolphine)
        self.imgs_COCO = list(self.captions_COCO.keys())
        random.shuffle(self.imgs_COCO)
        

        # split the dataset
        if split == 'train':
            self.imgs_turtle = self.imgs_turtle[ : int(0.8 * len(self.imgs_turtle))]
            self.imgs_debris = self.imgs_debris[: int(0.8 * len(self.imgs_debris))]
            self.imgs_sea = self.imgs_sea[: int(0.8 * len(self.imgs_sea))]
            self.imgs_dolphine = self.imgs_dolphine[: int(0.8 * len(self.imgs_dolphine))]
            self.imgs_COCO = self.imgs_COCO[ : int(0.5 * len(self.imgs_COCO))] # 20k
            #self.imgs_COCO = self.imgs_COCO[:20000]
        elif split == 'val':
            self.imgs_turtle = self.imgs_turtle[int(0.8 * len(self.imgs_turtle)) : int(0.9 * len(self.imgs_turtle))]
            self.imgs_debris = self.imgs_debris[int(0.8 * len(self.imgs_debris)) : int(0.9 * len(self.imgs_debris))]
            self.imgs_sea = self.imgs_sea[int(0.8 * len(self.imgs_sea)) : int(0.9 * len(self.imgs_sea))]
            self.imgs_dolphine = self.imgs_dolphine[int(0.8 * len(self.imgs_dolphine)) : int(0.9 * len(self.imgs_dolphine))]
            self.imgs_COCO = self.imgs_COCO[int(0.5 * len(self.imgs_COCO)) : int(0.65 * len(self.imgs_COCO))] # 16k
        elif split == "test":
            self.imgs_turtle = self.imgs_turtle[int(0.9 * len(self.imgs_turtle)) : ]
            '''self.imgs_debris = self.imgs_debris[int(0.9 * len(self.imgs_debris)) : ]
            self.imgs_sea = self.imgs_sea[int(0.9 * len(self.imgs_sea)) : ]
            self.imgs_dolphine = self.imgs_dolphine[int(0.9 * len(self.imgs_dolphine)) : ]'''
            self.imgs_COCO = self.imgs_COCO[int(0.9 * len(self.imgs_COCO)) : ]
        else: # use all images
            pass
        

        print("turtle: ", len(self.imgs_turtle))
        print("debris: ", len(self.imgs_debris))
        print("sea: ", len(self.imgs_sea))
        print("dolphine: ",len(self.imgs_dolphine))
        print("coco", len(self.imgs_COCO))
        
        #Create a unique list of keys
        self.imgs = self.imgs_COCO + self.imgs_turtle + self.imgs_debris + self.imgs_dolphine 
        random.shuffle(self.imgs)
        #Create a unique dictionary of captions
        self.captions = self.captions_COCO | self.captions_turtle | self.captions_debris | self.captions_sea | self.captions_dolphine
        
  
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
    
    


