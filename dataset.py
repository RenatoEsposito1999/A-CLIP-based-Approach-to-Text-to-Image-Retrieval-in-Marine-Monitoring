import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from transformers import CLIPTokenizer


'''
    MODIFICARE DATASET UTILIZZANDO LE CATEGORY DI COCO, DEBRIS E TURTLE E NO_TURTLE
'''

'''class RetrievalDataset(Dataset):
    def __init__(self, csv_path, transform_turtle=None, transform_coco= None, val_transform=None):
        self.df = pd.read_csv(csv_path)
        #self.transform = transform if transform else T.ToTensor()
        self.transform_turtle = transform_turtle
        self.transform_coco = transform_coco
        self.val_transform = val_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        caption = row['caption']

        if "Train_cropped" in row['image_path'] and self.val_transform is None:
           image = self.transform_turtle(image)
        elif "COCO" in row['image_path'] and self.val_transform is None:
            image = self.transform_coco(image)
        elif self.val_transform is not None:
            image = self.val_transform(image)
        else:
            image = T.ToTensor(image)
        
        return {
            'image': image,
            'caption': caption
        }'''
        
class RetrievalDataset(Dataset):
    def __init__(self, csv_path, transform_turtle=None, transform_coco= None, val_transform=None):
        self.df = pd.read_csv(csv_path)
        #self.transform = transform if transform else T.ToTensor()
        self.transform_turtle = transform_turtle
        self.transform_coco = transform_coco
        self.val_transform = val_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        turtle = None
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        caption = row['caption']

        if "Train_cropped" in row['image_path']:
            if self.val_transform is None:
                image = self.transform_turtle(image)
            else:
                image = self.val_transform(image)   
            turtle = 1
        elif "COCO" in row['image_path']:
            if self.val_transform is None:
                image = self.transform_coco(image)
            else:
                image = self.val_transform(image)
            turtle=0
        else:
            image = T.ToTensor(image)
        
        return {
            'image': image,
            'caption': caption,
            "turtle": turtle
        }
 
 
'''# --- Collate Function ---
def collate_fn(batch, tokenizer):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    return {
        'images': images,
        'captions': tokens
    }'''

# --- Collate Function ---
def collate_fn(batch, tokenizer):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    turtle = torch.tensor([item["turtle"] for item in batch])
    return {
        'images': images,
        'captions': tokens,
        "turtle": turtle
    }