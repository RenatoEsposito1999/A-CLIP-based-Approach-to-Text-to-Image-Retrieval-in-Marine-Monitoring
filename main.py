import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoProcessor, get_cosine_schedule_with_warmup
from custom_utils.telegram_notification import send_telegram_notification
from collections import defaultdict
from tqdm import tqdm
import torch
from src.sampler import ClassBalancedBatchSampler
from src.new_sampler import NonRepeatingBalancedSampler
from src.new_dataset import Custom_dataset_augmented
from src.nanoclip import NanoCLIP
from src.CLIP_model import CLIP_model
#from src.dataset import Custom_dataset, Collate_fn
from src.dataset_category_only_turtle import Custom_dataset_category_only_turtle, Collate_fn_nanoclip, Collate_fn_clip
from src.loss import contrastiveLoss
from src.train import train
import random
import numpy as np
import os
CHAT_ID_VINCENZO = "521260346"
CHAT_ID_RENATO = "407888332"

generic_ransform = T.Compose([
        T.Resize((224, 224)),
        #T.RandomRotation(15),
        #T.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=3),
        #T.RandomHorizontalFlip(0.5),
        #T.RandomVerticalFlip(0.1),
        #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.), # no hue because it distorts the colors
        #T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

'''train_heavy_transform = T.Compose([
    # Ridimensionamento e crop più aggressivo (range più ampio)
    #T.RandomResizedCrop(224, scale=(0.6, 1.2), ratio=(0.7, 1.3)),  # Scala e ratio più estremi
    
    # Flipping e rotazioni estreme
    T.RandomHorizontalFlip(p=0.7),  # Probabilità più alta
    T.RandomVerticalFlip(p=0.3),    # Aggiungi flip verticale
    T.RandomRotation(degrees=30),   # Rotazione fino a 30 gradi
    
    # Distorsioni prospettiche/geometriche
    T.RandomPerspective(distortion_scale=0.4, p=0.5),  # Effetto "warp"
    T.RandomAffine(degrees=0, translate=(0.2, 0.2)),   # Traslazioni casuali
    #T.ToTensor(),
    # Gaussian Blur di PyTorch
    #T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),   # Sostituisce il PIL ImageFilter
    
    # Alterazioni cromatiche pesanti
    T.ColorJitter(
        brightness=0.4,  # Variazione più forte di luminosità
        contrast=0.4,    # Contrasto più marcato
        saturation=0.3,  # Saturazione più variabile
        hue=0.1          # Tonalità più ampia (max consentito è 0.5)
    ),
    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])'''

train_heavy_transform = T.Compose([
    T.Resize((224, 224)),
    #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.),
    #T.ToTensor(),
    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def print_number_trainable_parameters(model):    
    total_params = sum(p.numel() for p in model.parameters())    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}\nTotal parameters: {total_params:,}")


def print_names_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Nome: {name} | Trainabile: {param.requires_grad} | Shape: {param.shape}")

def get_optimizer_and_scheduler(model,name_model, lr,weight_decay, tot_num_epochs, steps_per_epoch):
        """
        Define the optimizer and the learning rate scheduler.
        """
        
        #START DEFINITION OPTIMIZER
        if name_model == "nanoclip":
            trainable_params_img = [p for p in model.img_encoder.parameters() if p.requires_grad]
            trainable_params_text = [p for p in model.txt_encoder.parameters() if p.requires_grad]       
            optimizer_params = [
                {"params": trainable_params_img, "lr": lr, "weight_decay": weight_decay},
                {"params": trainable_params_text, "lr": lr, "weight_decay": weight_decay},
            ]
        elif name_model == "clip":
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer_params = [
                {"params": trainable_params, "lr": lr, "weight_decay": weight_decay},
            ]
        optimizer = torch.optim.AdamW(optimizer_params)
        
        #START DEFINITION COSINE SCHEDULER
        warmup_ratio = 0.1
        total_training_steps = tot_num_epochs * steps_per_epoch
        num_warmup_steps = int(warmup_ratio*total_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps
        )
        #milestones=[5, 10, 15]
        #lr_mult=0.1
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_mult)   
        
        return optimizer, scheduler

def get_class_with_index(dataset):
    class_to_indices = defaultdict(list)
    
    for idx, img_name in tqdm(enumerate(dataset.imgs), total=len(dataset.imgs)):
        category = dataset.captions[img_name][1]
        class_to_indices[category].append(idx)
    
    return dict(class_to_indices)


def get_next_version(log_dir: str) -> int:
    """Trova il prossimo numero di versione disponibile."""
    os.makedirs(log_dir, exist_ok=True)  # Crea la cartella se non esiste
    existing_versions = []
    
    for d in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("version_"):
            try:
                num = int(d.split("_")[1])
                existing_versions.append(num)
            except ValueError:
                continue  # Ignora cartelle con formato non valido
    
    return max(existing_versions) + 1 if existing_versions else 0

def seed_everything(seed: int):
    """Imposta il seed per tutte le librerie principali."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Per multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(batch_size, lr, dim, device, wd, name_model, n_epochs):
    # Cartella base per i log (es: "logs/NanoCLIP")
    seed_everything(12345)
    log_base_dir = "logs/NanoCLIP"
    next_version = get_next_version(log_base_dir)
    log_dir = os.path.join(log_base_dir, f"version_{next_version}")
    writer = SummaryWriter(log_dir=log_dir)  # Sostituisci "experiment_name" con un 
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name_model == "nanoclip":
        txt_model = "sentence-transformers/all-MiniLM-L6-v2" # (~22M params)
        
        model = NanoCLIP(
            txt_model=txt_model,
            img_model="dinov2_vits14",  # 'dinov2_vitb14' (60M params) or 'dinov2_vits14' (~22M params)
            unfreeze_n_blocks=4,        # unfreeze the last n blocks of both text and image encoders (for fine-tuning)
            embed_size=dim,             # output dimension of the encoders
            lr=lr,
            weight_decay=4e-4,
            warmup_epochs=5,
            milestones=[10, 20, 30],
            lr_mult=0.1,
        )
        print_number_trainable_parameters(model=model)
        tokenizer = AutoTokenizer.from_pretrained(txt_model)
        collate_fn = Collate_fn_nanoclip(tokenizer,  max_length=80, captions_to_use='first')
    elif name_model == "clip":
        model = CLIP_model()
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        collate_fn = Collate_fn_clip(processor=processor)
        print_number_trainable_parameters(model=model)
        #print_names_trainable_parameters(model=model)
        
    
    print("Train dataset")
    print("-"*15)
    #train_dataset = Custom_dataset('./datasets/', split='train', turtle_transform=train_heavy_transform, generic_transform= generic_ransform)
    #train_dataset = Custom_dataset_category_only_turtle('./datasets/', split='train', turtle_transform=train_heavy_transform, generic_transform= generic_ransform)
    train_dataset = Custom_dataset_augmented("./NEW_DATASET", split="train", model=name_model)
    print("-"*15)
    print("Validation dataset")
    #val_dataset = Custom_dataset('./datasets/', split='val', is_val=True)
    #val_dataset = Custom_dataset_category_only_turtle('./datasets/', split='val', is_val=True)
    val_dataset = Custom_dataset_augmented("./NEW_DATASET", split="val", model=name_model)
    print("-"*15)
    
    class_to_indices = get_class_with_index(train_dataset)
    #sampler = ClassBalancedBatchSampler(class_to_indices, batch_size=batch_size, classes_per_batch=16)
    train_sampler = NonRepeatingBalancedSampler(dataset=train_dataset, batch_size=batch_size)
    val_sampler = NonRepeatingBalancedSampler(dataset=val_dataset, batch_size=batch_size)
    train_dataloader = DataLoader(
        train_dataset, 
        #batch_size=batch_size, 
        #shuffle=True,
        batch_sampler = train_sampler, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_fn # captions_to_use='random' or 'first' or 'all'
    )

    val_dataloader = DataLoader(
        val_dataset, 
        #batch_size=batch_size, 
        #shuffle=False,
        batch_sampler = val_sampler,
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn # in eval we use the first caption only
    )
    
    print("Start training")
    optimizer, scheduler = get_optimizer_and_scheduler(model,name_model = name_model, lr=lr,weight_decay=wd, tot_num_epochs=n_epochs, steps_per_epoch=len(train_dataloader))
    send_telegram_notification(message="Inizio il Training!", CHAT_ID=CHAT_ID_VINCENZO)
    send_telegram_notification(message="Inizio il Training!", CHAT_ID=CHAT_ID_RENATO)
    train(model=model, dataloader=train_dataloader,n_epochs=n_epochs, loss_fn=contrastiveLoss,device=device,optimizer=optimizer,scheduler=scheduler, writer=writer, val_dataloader=val_dataloader)
    send_telegram_notification(message="Training completato!", CHAT_ID=CHAT_ID_VINCENZO)
    send_telegram_notification(message="Training completato!", CHAT_ID=CHAT_ID_RENATO)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train parameters")
    
    parser.add_argument("--dev", action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--bs", type=int, default=256, help="Batch size.")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimensionality.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--wd", type=float, default=4e-4, help="Weight decay")
    parser.add_argument("--model", type=str, default="clip", help="Model name")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epoch")

    args = parser.parse_args()
    
    main(batch_size=args.bs, lr=args.lr, dim=args.dim, device= args.device, wd = args.wd, name_model=args.model, n_epochs=args.n_epochs)