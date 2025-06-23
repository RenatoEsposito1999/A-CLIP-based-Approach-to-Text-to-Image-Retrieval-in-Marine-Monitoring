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

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = RetrievalModel(opts=opts).to(opts.device)
last_state = torch.load("/workspace/CLIP/text-to-image-retrivial/best_model_mix.pth")
model.load_state_dict(last_state)



# carica immagini
images = [Image.open(path) for path in ["/workspace/text-to-image-retrivial/DATASET/Train_cropped/cropped_frame_031810.PNG", 
                                        "/workspace/text-to-image-retrivial/DATASET/COCO/COCO_val2014_000000284851.jpg",
                                        "/workspace/text-to-image-retrivial/DATASET/COCO/COCO_val2014_000000248111.jpg",
                                        "/workspace/text-to-image-retrivial/DATASET/COCO/COCO_val2014_000000340529.jpg"]]

# caption
caption = "a cat and dog"

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


