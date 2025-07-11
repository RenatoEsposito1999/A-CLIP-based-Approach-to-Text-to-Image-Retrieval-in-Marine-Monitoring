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

CROPPED_TURTLE_POSITIVE_CSV_PATH = "./cropped_turtle.csv"
DATASET_IMAGES_CROPPED_PATH = "/workspace/text-to-image-retrivial/datasets/Turtle"
DATASET_ANNOTATIONS_PATH = "/workspace/annotations/instances_Train.json"


class Annotations:
    img_list = []
    ID = 0
    def __init__(self):
        self.LLM = None #self.LLM = LLM()
        with (open(CROPPED_TURTLE_POSITIVE_CSV_PATH, mode='w', encoding='utf-8', newline='')) as crop_turtle_file:
            fieldnames = ['image_name','caption']
            self.writer_crop_turtle = csv.DictWriter(crop_turtle_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            self.writer_crop_turtle.writeheader()

            #CREATE POSITIVE TURTLE CSV
            self.turtle_create_csv()
 
    def turtle_create_csv(self):
        for file_name in os.listdir(DATASET_IMAGES_CROPPED_PATH):
            dynamic_random = random.Random(time.time())
            mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
            sentence = generate_positive_sentence()
            if not mantain_templating and self.LLM:
                sentence = self.LLM.rephrase_sentence(sentence=sentence)
                
            self.writer_crop_turtle.writerow({"image_name": f"cropped_{file_name}", "caption": sentence})

    
dataset = Annotations()
