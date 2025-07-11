from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from tqdm import tqdm
import os
import csv
 
'''
This script is used in order generate captions by using blip and save into a csv file.
WARNING: can generate same captions, so after this you need to execute paraphraser.py
Use count different captions.py to validate the dataset. 
'''
output_csv = "Debris.csv"
input_folder = "/workspace/text-to-image-retrivial/datasets/Debris"  # Sostituisci con il tuo percorso
device = "cuda"
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
 
def process_images_batch(image_paths, batch_size=256):
    """Processa immagini in batch per ottimizzare l'uso della GPU"""
    results = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(img_path).convert("RGB") for img_path in batch_paths]
        
        # Preprocess batch
        inputs = processor(images, return_tensors="pt", padding=True).to(device)
        
        # Genera caption
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20)
        
        # Decodifica risultati
        captions = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Salva risultati
        for img_path, caption in zip(batch_paths, captions):
            results.append({
                "image_name": os.path.basename(img_path),
                "caption": caption.strip()
            })
            
    return results
 
# Main execution
if __name__ == "__main__":
    # Prendi tutti i file immagine dalla cartella
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    print(f"Trovate {len(image_paths)} immagini da processare")
    
    # Processa e salva i risultati
    results = process_images_batch(image_paths)
    
    # Scrivi CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["image_name", "caption"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Fatto! Risultati salvati in {output_csv}")
 
 
 