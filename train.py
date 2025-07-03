# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import AutoTokenizer

from src.nanoclip import NanoCLIP
#from src.dataset import Flickr30k, CollateFlickr
from src.custom_dataset_with_category import Custom_dataset, CollateFlickr
from custom_utils.telegram_notification import send_telegram_notification
from src.sampler import ClassBalancedBatchSampler
from tqdm import tqdm
from collections import defaultdict
import torch
CHAT_ID_VINCENZO = "521260346"
CHAT_ID_RENATO = "407888332"
'''def get_class_with_index(dataset):
    class_to_indices = defaultdict(list)
    for idx,item in enumerate(tqdm(dataset)):
        _,_,cat_id = item
        class_to_indices[cat_id].append(idx)
    return class_to_indices'''
def get_class_with_index(dataset):
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    
    for idx, img_name in tqdm(enumerate(dataset.imgs), total=len(dataset.imgs)):
        category = dataset.captions[img_name][1]
        class_to_indices[category].append(idx)
    
    return dict(class_to_indices)

        
def train(batch_size, lr, dim, dev):
    
    seed_everything(seed=20241203, workers=True)
    
    # txt_model = "albert-base-v2"
    # txt_model = "distilbert-base-uncased"
    # txt_model = "bert-base-uncased"
    txt_model = "sentence-transformers/all-MiniLM-L6-v2" # (~22M params)
    # txt_model = "sentence-transformers/all-MiniLM-L12-v2"
    # txt_model = "huawei-noah/TinyBERT_General_4L_312D"
    # txt_model = "sentence-transformers/all-mpnet-base-v2"
    # txt_model = "microsoft/MiniLM-L12-H384-uncased"
    
    
    # let's define the model.
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
    
    # data augmentation during training
    generic_ransform = T.Compose([
        #T.RandomRotation(15),
        #T.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=3),
        #T.RandomHorizontalFlip(0.5),
        #T.RandomVerticalFlip(0.1),
        #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.), # no hue because it distorts the colors
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    T.ToTensor(),
    # Gaussian Blur di PyTorch
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),   # Sostituisce il PIL ImageFilter
    
    # Alterazioni cromatiche pesanti
    T.ColorJitter(
        brightness=0.4,  # Variazione più forte di luminosità
        contrast=0.4,    # Contrasto più marcato
        saturation=0.3,  # Saturazione più variabile
        hue=0.1          # Tonalità più ampia (max consentito è 0.5)
    ),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    # no data augmentation during validation
    valid_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    '''train_dataset = Flickr30k('./datasets/flickr30k', split='train', img_transform=train_transform)
    val_dataset = Flickr30k('./datasets/flickr30k', split='val', img_transform=valid_transform)'''
    print("Train dataset")
    print("-"*15)
    train_dataset = Custom_dataset('./datasets/', split='train', turtle_transform=train_heavy_transform, generic_transform= generic_ransform)
    print("-"*15)
    print("Val dataset")
    val_dataset = Custom_dataset('./datasets/', split='val', generic_transform=generic_ransform, is_val=True)
    print("-"*15)
 
    # use the same tokenizer as the one used in the text model.
    tokenizer = AutoTokenizer.from_pretrained(txt_model)
    class_to_indices = get_class_with_index(train_dataset)
    sampler = ClassBalancedBatchSampler(class_to_indices, batch_size=batch_size, classes_per_batch=16)
    
    train_dataloader = DataLoader(
        train_dataset, 
        #batch_size=batch_size, 
        #shuffle=True,
        batch_sampler = sampler, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=CollateFlickr(tokenizer, max_length=80, captions_to_use='first') # captions_to_use='random' or 'first' or 'all'
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        collate_fn=CollateFlickr(tokenizer,  max_length=80, captions_to_use='first') # in eval we use the first caption only
    )
      
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"nano_clip",
        default_hp_metric=False
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor="category_all_r@5",
        filename="epoch=[{epoch:02d}]_recall@5=[{category_all_r@5:.4f}]]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )
    
    epoch_checkpoint_cb = ModelCheckpoint(
        filename="checkpoint",  # Formatta epoca e LR nel nome
        auto_insert_metric_name=False,  # Disabilita l'aggiunta automatica delle metriche
        save_weights_only=False,  # Salva solo i pesi (più leggero)
        every_n_epochs=1,  # Salva ogni epoca
        save_last=True,  # Salva sempre l'ultimo checkpoint (utile per il resume)
        save_top_k=-1,  # Salva tutti i checkpoint
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,      # comment this line if you don't want to use tensorboard logger
        precision="16-mixed",
        max_epochs=50,
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint_cb,# this callback saves the best model based on the metric we monitor (recall@5)
            epoch_checkpoint_cb,
            RichProgressBar()           # comment this line if you want classic progress bar
        ],
        #log_every_n_steps=10,
        fast_dev_run=dev,
        enable_model_summary=True,
    )
    print(len(train_dataloader))
    print("START TRAINING")
    #send_telegram_notification(message="Training iniziato!", CHAT_ID=CHAT_ID_VINCENZO)
    #send_telegram_notification(message="Training iniziato!", CHAT_ID=CHAT_ID_RENATO)
    trainer.fit(model, train_dataloader, val_dataloader)
    send_telegram_notification(message="Training completato!", CHAT_ID=CHAT_ID_VINCENZO)
    send_telegram_notification(message="Training completato!", CHAT_ID=CHAT_ID_RENATO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train parameters")
    
    parser.add_argument("--dev", action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--bs", type=int, default=256, help="Batch size.")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimensionality.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate.")
    args = parser.parse_args()
    
    train(batch_size=args.bs, lr=args.lr, dim=args.dim, dev=args.dev)
