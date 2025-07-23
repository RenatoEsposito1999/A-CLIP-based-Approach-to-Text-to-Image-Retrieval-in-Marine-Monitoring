import sys
sys.path.append("..")
import argparse
from transformers import AutoProcessor, pipeline
from PIL import Image
import torch
import numpy as np
from src.model import CLIP_model
import os
import librosa

def predict(model, processor, list_images_path, device, stt, path_audio):
    #Query text
    #query = "Turtle in the water"
    audio, sr = librosa.load(path_audio, sr=16000)  # Whisper richiede 16kHz
    # Trascrivi l'audio
    query = (stt(audio)["text"]).lower()
    images = [] #List of all images
    for img_path in list_images_path:
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
    #Apply similarity
    similarities = (text_embeds @ image_embeds.T).squeeze(0)  # shape: (num_images,)

    #Sort images by similarities
    sorted_indices = similarities.argsort(descending=True)
    print("Classification of images")
    for idx in sorted_indices:
        print(f"Image {idx} - Sim score: {similarities[idx].item():.4f}, PATH = {list_images_path[idx]}")


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

def get_pipeline_stt():
    stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3"
    )
    return stt_pipeline

def get_list_images(path_images):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_paths = []
    for root, _, files in os.walk(path_images):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    image_paths = [os.path.abspath(path) for path in image_paths]
    return image_paths

def main(device, model_name, path_images, path_audio):
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = get_model_processor(model_name=model_name, device=device)
    stt_pipeline = get_pipeline_stt()
    images_paths = get_list_images(path_images=path_images)
    predict(model=model, processor=processor, list_images_path=images_paths, device=device, stt=stt_pipeline, path_audio=path_audio)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train parameters")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    #laion/CLIP-ViT-B-32-laion2B-s34B-b79K
    #openai/clip-vit-base-patch32
    parser.add_argument("--model_name", type=str, default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", help="Pretrained model name")
    parser.add_argument("--path_images", type=str, default="./list_images", help="Pretrained model name")
    parser.add_argument("--audio_file", type=str, default="./audio.mp3", help="Path of the audio file")

    args = parser.parse_args()
    main(device=args.device,
         model_name=args.model_name,
         path_images=args.path_images,
         path_audio=args.audio_file)