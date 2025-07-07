import json
import random
import csv
from create_caption import generate_negative_sentence, generate_positive_sentence, generate_dolphine_sentence
import time
#from COCO_filter import get_caption
#from llm import LLM
import pandas as pd
from pycocotools.coco import COCO
import os
#from get_category_info import process_file
from collections import defaultdict

CAPTIONS_ANNOTATIONS_COCO_PATH = "/workspace/annotations/captions_val2014.json"
COCO_ISTANCES_VAL_PATH = "/workspace/annotations/instances_val2014.json"
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FLICKER_TXT = os.path.join(CURRENT_DIRECTORY, "captionsFlicker.txt")
CAPTIONS_COCO = COCO(CAPTIONS_ANNOTATIONS_COCO_PATH)        
INSTANCES_COCO = COCO(COCO_ISTANCES_VAL_PATH) 

LLM = None


def read_flicker_txt():
    flicker_dict = defaultdict(str)
    with open(FLICKER_TXT, 'r') as f:
        for line in f.readlines()[1:]: # ignore the header (first line)
            image, caption_number, caption = line.strip().split(',', 2)
            if caption_number == "0":
                flicker_dict[image] = caption
    return flicker_dict
                

def COCO_get_caption_and_category(img_name):
    #Trova l'immagine corrispondente
    for idx in CAPTIONS_COCO.imgs:
        if CAPTIONS_COCO.imgs[idx]["file_name"] == img_name:
            img_id = CAPTIONS_COCO.imgs[idx]["id"]
            ann_ids = CAPTIONS_COCO.getAnnIds(imgIds=img_id)
            captions = CAPTIONS_COCO.loadAnns(ann_ids)
            only_captions = []
            for caption in captions:
                only_captions.append(caption["caption"])
            caption = captions[random.randint(0,len(captions)-1)]['caption']
            obj_ann_ids = INSTANCES_COCO.getAnnIds(imgIds=img_id)
            obj_anns = INSTANCES_COCO.loadAnns(obj_ann_ids)
            category_ids = list(set([ann['category_id'] for ann in obj_anns]))
            if not category_ids:
                supercategory = "empty"
            else:
                supercategories = list(set([INSTANCES_COCO.cats[cat_id]['supercategory']
                                        for cat_id in category_ids]))
                supercategory = random.choice(supercategories)
                
            return random.choice(only_captions), supercategory

def COCO_flicker_create_csv(path_folder, flickr_dict):
    unique_category = 0
    coco_flicker_dataset_path = os.path.join(path_folder, "COCO-Flickr30k")
    coco_flicker_csv = os.path.join(path_folder, "Annotations/coco_flicker.csv")
    coco_flicker_file = open(coco_flicker_csv, mode='w', encoding='utf-8', newline='')
    fieldnames = ['image_name', 'comment_number','comment','category']
    writer_coco_flicker = csv.DictWriter(coco_flicker_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer_coco_flicker.writeheader()
    for file_name in os.listdir(coco_flicker_dataset_path):
        if "COCO" in file_name:
            # Ottieni la lista di caption e la categoria
            sentence, category = COCO_get_caption_and_category(file_name)
            if not category == "empty":
                sentence = sentence.replace("\n", "")
                #file_name = os.path.join(COCO_FLICKER_DATASET_PATH, file_name)
                writer_coco_flicker.writerow({"image_name": file_name, "comment_number": 0, "comment": sentence,"category":unique_category})
                unique_category += 1
            # Per ogni caption nella lista, scrivi una riga nel file .txt
            '''for i, caption in enumerate(captions):
                # Formato: image_name, comment_number, comment
                caption = caption.replace("\n", "")
                txt_file.write(f"{file_name}, {i}, {caption}\n")'''
            '''caption = caption.replace("\n", "")
            txt_file.write(f"{file_name}, {i}, {caption}\n")'''
        
        elif file_name in flickr_dict:
            #file_name = os.path.join(COCO_FLICKER_DATASET_PATH, file_name)
            writer_coco_flicker.writerow({"image_name": file_name, "comment_number": 0, "comment": flickr_dict[file_name],"category":unique_category})
            unique_category += 1
    coco_flicker_file.close()
    
def create_marine_csv(folder, writer, category):
    for file_name in os.listdir(folder):
        #file_name = os.path.join(folder, file_name)
        if category == -1: #dolphin
            dynamic_random = random.Random(time.time())
            mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
            sentence = generate_dolphine_sentence()
            if not mantain_templating and LLM:
                sentence = LLM.rephrase_sentence(sentence=sentence)
        elif category == -2: #turtle
            dynamic_random = random.Random(time.time())
            mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
            sentence = generate_positive_sentence()
            if not mantain_templating and LLM:
                sentence = LLM.rephrase_sentence(sentence=sentence)
        elif category == -3: #debris
            dynamic_random = random.Random(time.time())
            mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
            sentence = generate_negative_sentence(include_trash=True)
            if not mantain_templating and LLM:
                sentence = LLM.rephrase_sentence(sentence=sentence)
        elif category == -4: #sea
            dynamic_random = random.Random(time.time())
            mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
            sentence = generate_negative_sentence(include_trash=False)
            if not mantain_templating and LLM:
                sentence = LLM.rephrase_sentence(sentence=sentence) 
        writer.writerow({"image_name": file_name, "comment_number": 0, "comment": sentence,"category":category})
    

def create_turtle_debris_dolphin_sea_csv(path_folder):
    turtle_csv = os.path.join(path_folder, "Annotations/turtle.csv")
    turtle_file = open(turtle_csv, mode='w', encoding='utf-8', newline='')
    
    other_turtle_csv = os.path.join(path_folder, "Annotations/other_turtle.csv")
    other_turtle_file = open(other_turtle_csv, mode='w', encoding='utf-8', newline='')

    sea_csv = os.path.join(path_folder, "Annotations/sea.csv")
    sea_file = open(sea_csv, mode='w', encoding='utf-8', newline='')
    
    dolphin_csv = os.path.join(path_folder, "Annotations/dolphin.csv")
    dolphin_file = open(dolphin_csv, mode='w', encoding='utf-8', newline='')
    
    debris_csv = os.path.join(path_folder, "Annotations/debris.csv")
    debris_file = open(debris_csv, mode='w', encoding='utf-8', newline='')
    
    fieldnames = ['image_name', 'comment_number','comment','category']
    
    turtle_writer = csv.DictWriter(turtle_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    turtle_writer.writeheader()
    other_turtle_writer = csv.DictWriter(other_turtle_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    other_turtle_writer.writeheader()
    sea_writer = csv.DictWriter(sea_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    sea_writer.writeheader()
    dolphin_writer = csv.DictWriter(dolphin_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    dolphin_writer.writeheader()
    debris_writer = csv.DictWriter(debris_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    debris_writer.writeheader()
    

    turtle_dataset_path = os.path.join(path_folder, "Turtle")
    create_marine_csv(turtle_dataset_path, turtle_writer, -2)
    other_turtle_dataset_path = os.path.join(path_folder, "Turtle_other")
    create_marine_csv(other_turtle_dataset_path, other_turtle_writer, -2)
    sea_dataset_path =os.path.join(path_folder, "Sea")
    create_marine_csv(sea_dataset_path, sea_writer, -4)
    dolphin_dataset_path = os.path.join(path_folder, "Dolphin")
    create_marine_csv(dolphin_dataset_path, dolphin_writer, -1)
    debris_dataset_path = os.path.join(path_folder, "Debris")
    create_marine_csv(debris_dataset_path, debris_writer, -3)
    
    turtle_file.close()
    other_turtle_file.close()
    dolphin_file.close()
    sea_file.close()
    debris_file.close()
    


def main():
    train_path = os.path.join(CURRENT_DIRECTORY, "Train")
    val_path = os.path.join(CURRENT_DIRECTORY, "Validation")
    test_path = os.path.join(CURRENT_DIRECTORY, "Test")
    
    flickr_dict = read_flicker_txt()
    
    COCO_flicker_create_csv(train_path, flickr_dict)
    create_turtle_debris_dolphin_sea_csv(train_path)
    COCO_flicker_create_csv(val_path, flickr_dict)
    create_turtle_debris_dolphin_sea_csv(val_path)
    COCO_flicker_create_csv(test_path, flickr_dict)
    create_turtle_debris_dolphin_sea_csv(test_path)
    
    

if __name__ == "__main__":
    main()
    
            