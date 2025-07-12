import json
import random
import csv
from pycocotools.coco import COCO
import os

COCO_CSV_PATH = "/workspace/text-to-image-retrivial/datasets/annotations/COCO_with_category.csv"
COCO_DATASET_PATH = "/workspace/text-to-image-retrivial/datasets/COCO/"
CAPTIONS_ANNOTATIONS_COCO_PATH = "/workspace/text-to-image-retrivial/datasets/annotations/captions_val2014.json"
COCO_ISTANCES_VAL_PATH = "/workspace/text-to-image-retrivial/datasets/annotations/instances_val2014.json"


CAPTIONS_COCO = COCO(CAPTIONS_ANNOTATIONS_COCO_PATH)        
INSTANCES_COCO = COCO(COCO_ISTANCES_VAL_PATH) 

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

def COCO_create_csv():
    coco_file = open(COCO_CSV_PATH, mode='w', encoding='utf-8', newline='')
    fieldnames = ['image_name', 'caption', 'category']
    writer_coco = csv.DictWriter(coco_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer_coco.writeheader()        
    for file_name in os.listdir(COCO_DATASET_PATH):
        # Ottieni la lista di caption e la categoria
        sentence, category = COCO_get_caption_and_category(file_name)
        if not category == "empty":
            sentence = sentence.replace("\n", "")
            writer_coco.writerow({"image_name": file_name, "caption": sentence, "category":category})
    coco_file.close()
    
def main():
    COCO_create_csv()

if __name__ == '__main__':
    main()
    

    
     
    
