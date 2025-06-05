import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from transformers import CLIPTokenizer
import json
        
class RetrievalDataset(Dataset):
    def __init__(self, csv_path, transform_turtle=None, transform_coco= None, val_transform=None):
        self.df = pd.read_csv(csv_path)
        #self.transform = transform if transform else T.ToTensor()
        self.transform_turtle = transform_turtle
        self.transform_coco = transform_coco
        self.val_transform = val_transform
        with open('./dataset/category_info.json', 'r', encoding='utf-8') as json_file:
            self.category_dict = json.load(json_file)
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        caption = row['caption']
        category_id = self.category_dict[row['category']][0]

        if "Train_cropped" in row['image_path'] and self.val_transform is None:
            image = self.transform_turtle(image)
        elif "COCO" in row['image_path']and self.val_transform is None:
            image = self.transform_coco(image)
        else:
            image = self.val_transform(image)
            #image = T.ToTensor(image)
        
        return {
            'image': image,
            'caption': caption,
            'category_id': category_id
        }

# --- Collate Function ---
def collate_fn(batch, tokenizer):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")

    category_id = torch.tensor([item["category_id"] for item in batch])
    return {
        'images': images,
        'captions': tokens,
        'category_id': category_id
    }