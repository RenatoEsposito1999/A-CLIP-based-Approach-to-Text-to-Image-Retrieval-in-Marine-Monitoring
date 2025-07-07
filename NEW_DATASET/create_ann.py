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


COCO_FLICKER_DATASET_PATH = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/COCO-Flickr30k/"
TURTLE_DATASET_PATH = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/Turtle/"
OTHER_TURTLE_DATASET_PATH = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/Turtle_other/"
DEBRIS_DATASET_PATH = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/Debris/"
DOLPHINE_DATASET_PATH = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/Dolphin/"
SEA_DATASET_PATH = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/Sea/"
FLICKER_TXT = "./captionsFlicker.txt"
CAPTIONS_ANNOTATIONS_COCO_PATH = "/workspace/annotations/captions_val2014.json"
COCO_ISTANCES_VAL_PATH = "/workspace/annotations/instances_val2014.json"
DEBRIS_CSV = "./debris.csv"
DOLPHIN_CSV = "./dolphin.csv"
SEA_CSV = "./sea.csv"
COCO_FLICKER_CSV = "./coco_flicker.csv"
TURTLE_CSV = "./turtle.csv"
OTHER_TURTLE_CSV = "./other_turtle.csv"

CAPTIONS_COCO = COCO(CAPTIONS_ANNOTATIONS_COCO_PATH)        
INSTANCES_COCO = COCO(COCO_ISTANCES_VAL_PATH) 

LLM = None
unique_category = 0



def read_flicker_txt():
    flicker_dict = defaultdict(str)
    with open(FLICKER_TXT, 'r') as f:
        for line in f.readlines()[1:]: # ignore the header (first line)
            image, caption_number, caption = line.strip().split(',', 2)
            if caption_number == 0:
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

def COCO_flicker_create_csv():
    global unique_category
    flicker_txt = read_flicker_txt()
    coco_flicker_file = open(COCO_FLICKER_CSV, mode='w', encoding='utf-8', newline='')
    fieldnames = ['image_name', 'comment_number','comment','category']
    writer_coco_flicker = csv.DictWriter(coco_flicker_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    for file_name in os.listdir(COCO_FLICKER_DATASET_PATH):
        if "COCO" in file_name:
            # Ottieni la lista di caption e la categoria
            sentence, category = COCO_get_caption_and_category(file_name)
            if not category == "empty":
                sentence = sentence.replace("\n", "")
                file_name = os.path.join(COCO_FLICKER_DATASET_PATH, file_name)
                writer_coco_flicker.writerow({"image_name": file_name, "comment_number": 0, "comment": sentence,"category":unique_category})
                unique_category += 1
            # Per ogni caption nella lista, scrivi una riga nel file .txt
            '''for i, caption in enumerate(captions):
                # Formato: image_name, comment_number, comment
                caption = caption.replace("\n", "")
                txt_file.write(f"{file_name}, {i}, {caption}\n")'''
            '''caption = caption.replace("\n", "")
            txt_file.write(f"{file_name}, {i}, {caption}\n")'''
        elif file_name in flicker_txt:
            file_name = os.path.join(COCO_FLICKER_DATASET_PATH, file_name)
            writer_coco_flicker.writerow({"image_name": file_name, "comment_number": 0, "comment": flicker_txt[file_name],"category":unique_category})
            unique_category += 1
    coco_flicker_file.close()
    
def create_marine_csv(folder, writer, category):
    for file_name in os.listdir(folder):
        file_name = os.path.join(folder, file_name)
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
    

def create_turtle_debris_dolphin_sea_csv():
    turtle_file = open(TURTLE_CSV, mode='w', encoding='utf-8', newline='')
    other_turtle_file = open(OTHER_TURTLE_CSV, mode='w', encoding='utf-8', newline='')
    sea_file = open(SEA_CSV, mode='w', encoding='utf-8', newline='')
    dolphin_file = open(COCO_FLICKER_CSV, mode='w', encoding='utf-8', newline='')
    debris_file = open(DEBRIS_CSV, mode='w', encoding='utf-8', newline='')
    
    fieldnames = ['image_name', 'comment_number','comment','category']
    
    turtle_writer = csv.DictWriter(turtle_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    other_turtle_writer = csv.DictWriter(other_turtle_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    sea_writer = csv.DictWriter(sea_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    dolphin_writer = csv.DictWriter(dolphin_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    debris_writer = csv.DictWriter(debris_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    
    create_marine_csv(TURTLE_DATASET_PATH, turtle_writer, -2)
    create_marine_csv(OTHER_TURTLE_DATASET_PATH, other_turtle_writer, -2)
    create_marine_csv(SEA_DATASET_PATH, sea_writer, -4)
    create_marine_csv(DOLPHINE_DATASET_PATH, dolphin_writer, -1)
    create_marine_csv(DEBRIS_DATASET_PATH, debris_writer, -3)
    
    turtle_file.close()
    other_turtle_file.close()
    dolphin_file.close()
    sea_file.close()
    debris_file.close()
    
COCO_flicker_create_csv()
create_turtle_debris_dolphin_sea_csv()
    
    
    
            