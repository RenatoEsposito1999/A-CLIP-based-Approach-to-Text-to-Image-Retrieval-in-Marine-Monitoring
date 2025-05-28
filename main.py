import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from transformers import CLIPTokenizer
from dataset import RetrievalDataset, collate_fn
from model import RetrievalModel
from loss import contrastive_loss


# --- Tokenizer ---
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
 
# --- Transforms ---
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
 
# --- Dataset and Dataloader ---
train_dataset = RetrievalDataset("train.csv", transform=image_transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
 
# --- Model and Optimizer ---
model = RetrievalModel(text_encoder="clip").cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
 
# --- Training Loop ---
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images = batch["images"].cuda()
        text_inputs = {k: v.cuda() for k, v in batch["captions"].items()}
 
        image_embeds, text_embeds = model(images, text_inputs)
        loss = contrastive_loss(image_embeds, text_embeds)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")