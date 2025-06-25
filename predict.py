import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoProcessor
from transformers import get_cosine_schedule_with_warmup
from dataset import RetrievalDataset, collate_fn
#from model import RetrievalModel
from model_only_clip import RetrievalModel
from loss import supcon_loss
from opts import parse_opts
from train import Train
import os
import shutil
from custom_utils.telegram_notification import send_telegram_notification
from PIL import Image
opts = parse_opts()

processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
model = RetrievalModel(opts=opts).to(opts.device)
last_state = torch.load(opts.best_model_mix_path)
model.load_state_dict(last_state)



# carica immagini
images = [Image.open(path) for path in ["/workspace/COCO/COCO_val2014_000000217886.jpg", 
                                        "/workspace/COCO/COCO_val2014_000000119709.jpg",
                                        "/workspace/Train_cropped/cropped_frame_054807.PNG",
                                        "/workspace/COCO/COCO_val2014_000000230838.jpg",
                                        "/workspace/COCO/COCO_val2014_000000037470.jpg"]]

# caption
caption = "a cat plays"

# usa il processor per ottenere i tensori
inputs = processor(
    text=[caption], 
    images=images, 
    return_tensors="pt", 
    padding=True
)

with torch.no_grad():
    image_embeds, text_embeds, logit_scale = model(
        images=inputs["pixel_values"].to(opts.device), 
        text_inputs=inputs["input_ids"].to(opts.device), 
        attention_mask=inputs["attention_mask"].to(opts.device)
    )

# text_embeds: (1, D), image_embeds: (3, D)
similarities = (text_embeds @ image_embeds.T).squeeze()  # (3,)
print(similarities)
most_similar_idx = similarities.argmax().item()
print(f"Immagine più simile: {most_similar_idx} con similarità {similarities[most_similar_idx].item():.4f}")


