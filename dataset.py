import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from transformers import CLIPTokenizer


class RetrievalDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform if transform else T.ToTensor()
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        caption = row['caption']
 
        if self.transform:
            image = self.transform(image)
 
        return {
            'image': image,
            'caption': caption
        }
 
# --- Collate Function ---
def collate_fn(batch, tokenizer):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    return {
        'images': images,
        'captions': tokens
    }