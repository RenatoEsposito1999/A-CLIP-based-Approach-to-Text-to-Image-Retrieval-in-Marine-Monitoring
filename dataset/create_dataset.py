import sys
sys.path.append('../blip2_tuned')

from dataset import ClipDataset
from datasets import load_dataset, concatenate_datasets
import random
from define_model_processor import load_processor
from torchvision import transforms


# Dimensione dell'immagine (es. CLIP usa solitamente 224x224)
IMAGE_SIZE = 224

# Trasformazioni per il training
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),  # applicabile se semanticamente valido
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])

# Trasformazioni per la validation
val_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    #transforms.CenterCrop(IMAGE_SIZE),
])


'''def create_dataset(opts, pos_path, neg_path, coco_path):
    processor = load_processor()
    dataset_pos = load_dataset("csv", data_files=pos_path, split=f"train[:{opts.pos_size}]")
    dataset_neg = load_dataset("csv", data_files=neg_path, split=f"train[:{opts.neg_size}]")
    dataset_COCO = load_dataset("csv", data_files=coco_path, split="train")
    

    # Positive split
    pos_train = dataset_pos.select(range(0, int(0.7 * len(dataset_pos))))
    pos_val   = dataset_pos.select(range(int(0.7 * len(dataset_pos)), int(0.85 * len(dataset_pos))))
    pos_test  = dataset_pos.select(range(int(0.85 * len(dataset_pos)), len(dataset_pos)))

    #COCO SPLIT
    coco_train = dataset_COCO.select(range(0, int(0.8 * len(dataset_COCO))))
    coco_val =  dataset_COCO.select(range(int(0.8 * len(dataset_COCO)), len(dataset_COCO)))
    # Negative split
    neg_train = dataset_neg.select(range(0, int(0.7 * len(dataset_neg))))
    neg_val   = dataset_neg.select(range(int(0.7 * len(dataset_neg)), int(0.85 * len(dataset_neg))))
    neg_test  = dataset_neg.select(range(int(0.85 * len(dataset_neg)), len(dataset_neg)))

    # Concat and shuffle
    #train_dataset = concatenate_datasets([pos_train, neg_train, coco_train]).shuffle(seed=42)
    #val_dataset   = concatenate_datasets([pos_val, neg_val,coco_val]).shuffle(seed=42)
    train_dataset = concatenate_datasets([pos_train, neg_train]).shuffle(seed=42)
    val_dataset   = concatenate_datasets([pos_val, neg_val]).shuffle(seed=42)
    test_dataset  = concatenate_datasets([pos_test, neg_test]).shuffle(seed=42)

    print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

    opts.gradient_accumulation_steps = 2
    steps_per_epoch = len(train_dataset) // (opts.batch_size * opts.gradient_accumulation_steps)
    total_steps = steps_per_epoch * opts.n_epochs
    print(f"Total expected training steps: {total_steps}")
        
    train_dataset = ClipDataset(dataset=train_dataset, processor=processor, transform=train_transforms)
    val_dataset = ClipDataset(dataset=val_dataset, processor=processor, transform=val_transforms)
    test_dataset = ClipDataset(dataset=test_dataset, processor=processor, transform=val_transforms)
    return train_dataset, val_dataset, test_dataset'''
    
    
def create_dataset(processor):
    #train_dataset = load_dataset("csv", data_files="../dataset/train.csv", split=f"train")
    #val_dataset = load_dataset("csv", data_files="../dataset/cropped_turtle_val.csv", split=f"train")
    train_dataset = load_dataset("csv", data_files="../dataset/COCO_train.csv", split=f"train")
    val_dataset = load_dataset("csv", data_files="../dataset/COCO_val.csv", split=f"train")
    
    train_dataset = ClipDataset(dataset=train_dataset, processor=processor, transform=train_transforms)
    val_dataset = ClipDataset(dataset=val_dataset, processor=processor, transform=val_transforms)
    
    return train_dataset, val_dataset
    