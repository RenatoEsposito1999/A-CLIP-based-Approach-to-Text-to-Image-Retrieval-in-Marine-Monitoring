import sys
sys.path.append('../../shared')


from torch.utils.data import Dataset
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict
from CONST import *
import torch
import random
import re

# RIMUOVERE CARATTERI STRANI E CONSIDERARE LE CAPTION TUTTE IN PICCOLO, forse è da fare anche nel testing. 

def clean_generated_caption(caption: str) -> str:
    # Rimuove caratteri non stampabili e sostituisce char corrotti
    caption = caption.encode("utf-8", "replace").decode("utf-8").replace("�", "")
    caption = re.sub(
        r"[^\x20-\x7E]", "", caption
    )  # Rimuove caratteri non ASCII visibili
    return caption.strip().lower()


class ClipDataset(Dataset):
    def __init__(self, dataset, processor, transform = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_name = item["image_path"]
        img = Image.open(item["image_path"])

        if self.transform:
            img = self.transform(img)
        
        caption = clean_generated_caption(item["caption"])

        return {
            "image": img,
            "text": caption,
            "image_path": image_name 
        }
    
@dataclass
class ClipCollator:
    processor: any  #Processor for processing image and text
    test = False
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        text = [item["text"] for item in batch]
        
        encoding = self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding="longest",
        )
        
        #encoding["labels"] = encoding["input_ids"].clone()
        
        if self.test:
            encoding["image_path"] = [item["image_path"] for item in batch]
        

        return encoding

