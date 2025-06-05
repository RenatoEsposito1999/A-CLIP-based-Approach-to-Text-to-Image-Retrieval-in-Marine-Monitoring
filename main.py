import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from transformers import AutoTokenizer
from dataset import RetrievalDataset, collate_fn
from model import RetrievalModel
from loss import contrastive_loss
from opts import parse_opts
from train import Train
import os
import shutil

def print_trainable_parameters(model):    
    total_params = sum(p.numel() for p in model.parameters())    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}\nTotal parameters: {total_params:,}")
    #return trainable_params, total_params

if __name__ == "__main__":
    opts = parse_opts()

    # --- Checkpoint directory check ---
    if not os.path.exists(opts.resume_path):
        os.makedirs(opts.resume_path)

    if not os.path.exists(opts.metrics_path):
        os.makedirs(opts.metrics_path)
    elif os.path.exists(opts.metrics_path) and not opts.resume: # if the path exists and resume is false then the path refers to old metre, so I delete the files inside
        # Deletes ALL contents of the directory (files and subdirectories)
        shutil.rmtree(opts.metrics_path)
        # Recreate the empty directory
        os.makedirs(opts.metrics_path)


    # --- Tokenizer ---
    #clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = AutoTokenizer.from_pretrained(opts.text_encoder)
    
    # --- Transforms ---
    # N.B Both transforms are equal, but in future we could apply arg. 
    train_coco_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711))
])
    train_turtle_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711))
])
    val_image_transform = T.Compose([
    T.Resize((224, 224)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711))
])
    
    train_image_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    val_image_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    

    # Model and optimizer
    model = RetrievalModel(opts=opts).to(opts.device)
    print_trainable_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    
    if not opts.no_train:
        # --- Dataset and Dataloader ---
        train_dataset = RetrievalDataset(opts.dataset_path, transform_turtle=train_image_transform, transform_coco = train_coco_transform)
        val_dataset = RetrievalDataset(opts.validation_path,val_transform=val_image_transform)
        only_turtle_val_dataset = RetrievalDataset(opts.only_turtle_validation_path,transform_turtle=val_image_transform)
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
        val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
        only_turtle_val_loader=None
        #only_turtle_val_loader = DataLoader(only_turtle_val_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
        print("START TRAINING")
        # --- Model and Optimizer ---
        
        trainer = Train(model=model,loss_fn=contrastive_loss,optimizer=optimizer, opts=opts)
        trainer.train_loop(train_loader=train_loader,val_loader=val_loader,only_turtle_loader=only_turtle_val_loader)
        
    elif opts.test:
        print("TO DO TEST")
    
