import sys
sys.path.append("../src")
import argparse
from transformers import AutoProcessor
from PIL import Image
import torch
import numpy as np
from model import CLIP_model
import os

def predict(model, processor, list_images_path, device):
    # Query testuale
    query = "Aerial view of turtle in the ocean"
    images = []
    for img_path in list_images_path:
    # Lista immagini
        images.append(Image.open(img_path))
    
    inputs = processor(text=[query], images=images, return_tensors="pt", padding=True)
    text_inputs = inputs["input_ids"].to(device)
    attention_masks=inputs["attention_mask"].to(device)
    images=inputs["pixel_values"].to(device)
    with torch.no_grad():
        image_embeds, text_embeds, _ = model(
            text_inputs=text_inputs,
            attention_masks=attention_masks,
            images=images
        )
    # Similarità coseno
    similarities = (text_embeds @ image_embeds.T).squeeze(0)  # shape: (num_images,)

    # Ordina le immagini per similarità
    sorted_indices = similarities.argsort(descending=True)
    print("Classifica immagini:")
    for idx in sorted_indices:
        print(f"Immagine {idx} - Similarità: {similarities[idx].item():.4f}, PATH = {list_images_path[idx]}")


def get_model_processor(model_name, device):
    #DEFINE THE MODEL CLIP
    model = CLIP_model(model_name=model_name)
    #processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    processor = AutoProcessor.from_pretrained(model_name)
    if "laion" in model_name:
        state = torch.load("../best_recall@5_focus_LAION.pth")
        model.load_state_dict(state["state_dict"])
    elif "openai" in model_name:
        state = torch.load("../best_recall@5_focus_OpenAI.pth")
        model.load_state_dict(state["state_dict"])
    model = model.to(device)
    return model, processor

def get_list_images(path_images):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_paths = []
    for root, _, files in os.walk(path_images):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    image_paths = [os.path.abspath(path) for path in image_paths]
    return image_paths

def main(device, model_name, path_images):
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = get_model_processor(model_name=model_name, device=device)
    images_paths = get_list_images(path_images=path_images)
    predict(model=model, processor=processor, list_images_path=images_paths, device=device)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train parameters")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    #laion/CLIP-ViT-B-32-laion2B-s34B-b79K
    #openai/clip-vit-base-patch32
    parser.add_argument("--model_name", type=str, default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", help="Pretrained model name")
    parser.add_argument("--path_images", type=str, default="./list_images", help="Pretrained model name")
    args = parser.parse_args()
    main(device=args.device,
         model_name=args.model_name,
         path_images=args.path_images)