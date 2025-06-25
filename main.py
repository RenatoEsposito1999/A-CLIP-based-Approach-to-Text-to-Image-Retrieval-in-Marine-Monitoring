import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoProcessor
from transformers import get_cosine_schedule_with_warmup
from dataset import RetrievalDataset, collate_fn
#from model import RetrievalModel
from model_only_clip import RetrievalModel
from loss import supcon_loss, masked_contrastive_loss
from opts import parse_opts
from train import Train
import os
import shutil
from custom_utils.telegram_notification import send_telegram_notification
from test import tester
from PIL import ImageFilter
from sampler import BalancedBatchSampler
import json
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
CHAT_ID_VINCENZO = "521260346"
CHAT_ID_RENATO = "407888332"

'''T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711))'''

def get_labels_list(opts):
    # 1. Carica il JSON delle categorie
    with open('./dataset/category_info.json') as f:
        categories = json.load(f)

    # 2. Leggi il CSV e estrai gli ID numerici
    category_ids = []
    with open(opts.dataset_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            category_name = row['category']
            category_id = categories[category_name][0]  # Prendi il primo elemento della lista (ID numerico)
            category_ids.append(category_id)
    
    return category_ids

def print_trainable_parameters(model):    
    total_params = sum(p.numel() for p in model.parameters())    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}\nTotal parameters: {total_params:,}")
    #return trainable_params, total_params
    
def set_scheduler(optimizer, tot_num_epochs, steps_per_epoch):
    warmup_ratio = 0.1
    total_training_steps = tot_num_epochs * steps_per_epoch
    num_warmup_steps = int(warmup_ratio*total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    return scheduler
    
    
if __name__ == "__main__":
    
    opts = parse_opts()

    if opts.device != 'cpu':
        opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Checkpoint directory check ---
    if not os.path.exists(opts.resume_path):
        os.makedirs(opts.resume_path)

    if not os.path.exists(opts.metrics_path):
        os.makedirs(opts.metrics_path)
    '''elif os.path.exists(opts.metrics_path) and not opts.resume and not opts.no_train: # if the path exists and resume is false then the path refers to old metre, so I delete the files inside
        # Deletes ALL contents of the directory (files and subdirectories)
        shutil.rmtree(opts.metrics_path)
        # Recreate the empty directory
        os.makedirs(opts.metrics_path)'''


    # --- Tokenizer ---
    #clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    #processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    
    # --- Transforms ---
    # train_coco_transform is empty for val dataset. 
    train_coco_transform = T.Compose([
    ])
    train_turtle_transform = T.Compose([
    T.RandomResizedCrop(224,scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.RandomRotation(degrees=10),
])
    
    train_heavy_transform = T.Compose([
    # Ridimensionamento e crop più aggressivo (range più ampio)
    T.RandomResizedCrop(224, scale=(0.6, 1.2), ratio=(0.7, 1.3)),  # Scala e ratio più estremi
    
    # Flipping e rotazioni estreme
    T.RandomHorizontalFlip(p=0.7),  # Probabilità più alta
    T.RandomVerticalFlip(p=0.3),    # Aggiungi flip verticale
    T.RandomRotation(degrees=30),   # Rotazione fino a 30 gradi
    
    # Distorsioni prospettiche/geometriche
    T.RandomPerspective(distortion_scale=0.4, p=0.5),  # Effetto "warp"
    T.RandomAffine(degrees=0, translate=(0.2, 0.2)),   # Traslazioni casuali
    # Applica GaussianBlur come filtro PIL
    lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)),  # Radius controlla l'intensità
    # Alterazioni cromatiche pesanti
    T.ColorJitter(
        brightness=0.4,  # Variazione più forte di luminosità
        contrast=0.4,    # Contrasto più marcato
        saturation=0.3,  # Saturazione più variabile
        hue=0.1          # Tonalità più ampia (max consentito è 0.5)
    ),
])


    # Model and optimizer
    model = RetrievalModel(opts=opts).to(opts.device)
    print_trainable_parameters(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Parametri ottimizzati:")
    optimizer = torch.optim.AdamW(trainable_params, lr=opts.learning_rate, weight_decay=opts.weight_decay)
    
    if not opts.no_train:
        # --- Dataset and Dataloader ---
        labels = get_labels_list(opts=opts)
        sampler = BalancedBatchSampler(batch_size=opts.batch_size, labels=labels,n_classes=16)
        
        train_dataset = RetrievalDataset(opts.dataset_path, transform_turtle=train_heavy_transform, transform_coco = train_coco_transform)
        
        val_dataset = RetrievalDataset(opts.validation_path,val_transform=train_coco_transform)
        #train_loader = DataLoader(train_dataset, batch_size=opts.batch_size,num_workers=4,  shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
        train_loader = DataLoader(train_dataset,num_workers=4, batch_size=opts.batch_size, sampler=sampler, collate_fn=lambda b: collate_fn(b, processor))
        val_loader = DataLoader(val_dataset, batch_size=opts.batch_size,num_workers=4, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
        #scheduler = set_scheduler(optimizer=optimizer, tot_num_epochs=opts.n_epochs, steps_per_epoch=len(train_loader))
        scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',       # Monitora il val_loss (minimizzare)
        factor=0.1,      # Riduci il LR di 10x quando si attiva
        patience=3,      # Aspetta 3 epoche senza miglioramenti
        verbose=True     # Stampa un messaggio quando il LR viene ridotto
        )
        print("START TRAINING")
        send_telegram_notification(message="Training iniziato!", CHAT_ID=CHAT_ID_VINCENZO)
        send_telegram_notification(message="Training iniziato!", CHAT_ID=CHAT_ID_RENATO)
        # --- Model and Optimizer ---
        #trainer = Train(model=model,loss_fn=contrastive_loss,optimizer=optimizer, opts=opts)
        trainer = Train(model=model,loss_fn=masked_contrastive_loss,optimizer=optimizer,scheduler = scheduler, opts=opts)
        trainer.train_loop(train_loader=train_loader,val_loader=val_loader)
        send_telegram_notification(message="Training completato!", CHAT_ID=CHAT_ID_VINCENZO)
        send_telegram_notification(message="Training completato!", CHAT_ID=CHAT_ID_RENATO)
    elif opts.test:
        test_dataset = RetrievalDataset(opts.test_path, transform_turtle=train_coco_transform, transform_coco = train_coco_transform)
        test_loader = DataLoader(test_dataset, batch_size=opts.batch_size,num_workers=4, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
        tester = tester(opts=opts, model=model, loss_fn = supcon_loss, test_loader = test_loader)
        tester.test()
        print("TO DO TEST")
    
