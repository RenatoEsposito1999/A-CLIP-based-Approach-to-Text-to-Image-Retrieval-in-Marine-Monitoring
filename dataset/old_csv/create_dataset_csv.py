import json
import random
import csv
from create_caption import generate_negative_sentence, generate_positive_sentence, generate_dolphine_sentence
import time
#from COCO_filter import get_caption
#from custom_utils.llm import LLM
import pandas as pd
from pycocotools.coco import COCO
import os

DATASET_IMAGES_CROPPED_PATH = "/projects/data/turtle-full-v2/images/Train_cropped/"
#DATASET_IMAGES_PATH = "/projects/data/turtle-full-v2/images/Train/"
DATASET_ANNOTATIONS_PATH = "/projects/data/turtle-full-v2/annotations/instances_Train.json"
CROPPED_TURTLE_TRAIN_CSV_PATH= "cropped_turtle_train.csv"
CROPPED_TURTLE_VAL_CSV_PATH= "cropped_turtle_val.csv"
CROPPED_TURTLE_TEST_CSV_PATH = "cropped_turtle_test.csv"
CROPPED_TURTLE_POSITIVE_CSV_PATH = "all_cropped_turtle_positive.csv"

COCO_ISTANCES_VAL_PATH = "/projects/data/turtle-full-v2/annotations/instances_val2014.json"

CAPTIONS_ANNOTATIONS_COCO_PATH = "/projects/data/turtle-full-v2/annotations/captions_val2014.json"
COCO_DATASET_PATH="/projects/data/turtle-full-v2/COCO/"
COCO_CSV_PATH = "COCO_all.csv"
COCO_TRAIN_CSV_PATH = "COCO_train.csv"
COCO_VAL_CSV_PATH = "COCO_val.csv"
COCO_TEST_CSV_PATH = "COCO_test.csv"

FINAL_TRAIN_CSV_PATH = "train.csv"
FINAL_VAL_CSV_PATH = "val.csv"
#NEGATIVE_CSV_WITH_TRASH = "all_negative_with_trash.csv"
#NEGATIVE_CSV_WITHOUT_TRASH = "all_negative_without_trash.csv"
#COCO_DATASET_PATH="/projects/data/turtle-full-v2/COCO/"

class Annotations:
    img_list = []
    def __init__(self,train_size,val_size,nTrainPos,nTrainNeg,nValPos,nValNeg):
        self.img_list = self.extract_img()
        self.captions_coco = COCO(CAPTIONS_ANNOTATIONS_COCO_PATH)
        self.instances_coco = COCO(COCO_ISTANCES_VAL_PATH)    
        self.train_size = train_size
        self.val_size = val_size
        self.nTrainPos = nTrainPos
        self.nTrainNeg = nTrainNeg
        self.nValPos = nValPos
        self.nValNeg = nValNeg
        self.total_pos = nTrainPos + nValPos
        self.total_neg = nTrainNeg + nValNeg
        with open("info_dataset.txt", "w") as file:
            file.write("=== LOG DATASET ===\n\n")
        self.LLM = None #self.LLM = LLM()
        
        #CREATE POSITIVE TURTLE CSV
        #self.turtle_create_csv()
        
        #CREATE NEGATIVE COCO CSV
        #self.COCO_create_csv()
        
        #SPLIT TURTLE TRAIN AND VAL TURTLE
        #self.split_csv(input_file=CROPPED_TURTLE_POSITIVE_CSV_PATH, train_output=CROPPED_TURTLE_TRAIN_CSV_PATH, val_output=CROPPED_TURTLE_VAL_CSV_PATH,test_output=CROPPED_TURTLE_TEST_CSV_PATH, num_sample=10000)
        
        #SPLIT COCO TRAIN AND VAL
        #self.split_csv(input_file=COCO_CSV_PATH, train_output=COCO_TRAIN_CSV_PATH, val_output=COCO_VAL_CSV_PATH,test_output=COCO_TEST_CSV_PATH, num_sample=40000)
        
        #MERGE TRAIN OF TURTLE AND COCO IN ONE TRAIN
        #self.merge_and_shuffle_csv(CROPPED_TURTLE_TRAIN_CSV_PATH, COCO_TRAIN_CSV_PATH, FINAL_TRAIN_CSV_PATH)
        #MERGE VAL OF TURTEL AND COCO IN ONE VAL
        #self.merge_and_shuffle_csv(CROPPED_TURTLE_VAL_CSV_PATH, COCO_VAL_CSV_PATH, FINAL_VAL_CSV_PATH)
        
        #self.buildCSV()
        self.COCO_create_csv()
        #self.build_training_validation_CSV()
        
    
    def log_to_file(self,message, mode="a"):
        with open("info_dataset.txt", mode) as file:
            file.write(message + "\n")  # Aggiunge un ritorno a capo


    def extract_img(self):
        with open(DATASET_ANNOTATIONS_PATH) as file_json:
            annotations = json.load(file_json)
        for item_annotations in annotations['annotations']: 
            for item_img in annotations['images']:
                if item_img['id'] == item_annotations['image_id']:
                    self.img_list.append((item_img['file_name'], item_annotations['category_id']))
        random.shuffle(self.img_list)
        return self.img_list
    
    def turtle_create_csv(self):
        with open(CROPPED_TURTLE_POSITIVE_CSV_PATH,'w', newline='') as csv_file:
            fieldnames = ['image_path', 'caption']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            counter_positive = 0
            writer.writeheader()
            for img_name in self.img_list:
                if img_name[1] == 2: #img with turt == 2
                    dynamic_random = random.Random(time.time())
                    mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                    sentence = generate_positive_sentence()
                    if not mantain_templating and self.LLM:
                        sentence = self.LLM.rephrase_sentence(sentence=sentence)
                    writer.writerow({'image_path': DATASET_IMAGES_CROPPED_PATH+"cropped_"+img_name[0], 'caption':sentence})
                    counter_positive += 1
                #if counter_positive >= self.total_pos:
                    #break
            self.log_to_file(f"Total Positive data: {counter_positive}")
        
    def buildCSV(self):
        with open("cropped_marine_dataset.csv",'w', newline='') as csv_file:
            fieldnames = ['image_path', 'caption','category']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            #counter_negative_with_trash = 0
            writer.writeheader()
            for img_name in self.img_list:
                if img_name[1] == 1: #img with dolphine
                    dynamic_random = random.Random(time.time())
                    mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                    sentence = generate_dolphine_sentence()
                    if not mantain_templating and self.LLM:
                        sentence = self.LLM.rephrase_sentence(sentence=sentence)
                    writer.writerow({'image_path': DATASET_IMAGES_CROPPED_PATH+"cropped_"+img_name[0], 'caption':sentence, 'category':'dolphine'})
                if img_name[1] == 2: #img with turt == 2
                    dynamic_random = random.Random(time.time())
                    mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                    sentence = generate_positive_sentence()
                    if not mantain_templating and self.LLM:
                        sentence = self.LLM.rephrase_sentence(sentence=sentence)
                    writer.writerow({'image_path': DATASET_IMAGES_CROPPED_PATH+"cropped_"+img_name[0], 'caption':sentence, 'category':'turtle'})
                if img_name[1] == 3: ## img with trash
                    dynamic_random = random.Random(time.time())
                    mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                    sentence = generate_negative_sentence(include_trash=True)
                    if not mantain_templating and self.LLM:
                        sentence = self.LLM.rephrase_sentence(sentence=sentence)
                    writer.writerow({'image_path': DATASET_IMAGES_CROPPED_PATH+"cropped_"+img_name[0], 'caption':sentence, 'category':'debris'})
                elif img_name[1] == 4: #other are sea view
                    dynamic_random = random.Random(time.time())
                    mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                    sentence = generate_negative_sentence(include_trash=False)
                    if not mantain_templating and self.LLM:
                        sentence = self.LLM.rephrase_sentence(sentence=sentence)
                    writer.writerow({'image_path': DATASET_IMAGES_CROPPED_PATH+"cropped_"+img_name[0], 'caption':sentence, 'category':'sea'})
                '''if counter_negative >= self.total_neg:
                    break'''
    
    def split_csv(self, input_file, train_output, val_output, test_output, shuffle=True, train_ratio=0.8,val_ratio=0.1, num_sample=10000):
        df = pd.read_csv(input_file)
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.iloc[:num_sample]

        # Calcola i punti di split per train, validation e test
        train_split = int(len(df) * train_ratio)
        val_split = train_split + int(len(df) * val_ratio)  # Se val_ratio è dato
        # Oppure, se vuoi solo train_ratio e il resto è diviso tra val e test:
        test_split = int(len(df) * (train_ratio + val_ratio))  # Alternativa

        # Divisione in train, validation e test
        train_df = df.iloc[:train_split]
        val_df = df.iloc[train_split:test_split]  # Se usi val_ratio e test_ratio
        test_df = df.iloc[test_split:]  # Il resto va al test set

        # Salvataggio dei CSV
        train_df.to_csv(train_output, index=False)
        val_df.to_csv(val_output, index=False)
        test_df.to_csv(test_output, index=False)  # Nuovo file di test

        self.log_to_file(f"Divisione completata:")
        self.log_to_file(f"- Training set: {len(train_df)} righe ({train_ratio*100:.0f}%) salvato in {train_output}")
        self.log_to_file(f"- Validation set: {len(val_df)} righe ({(val_ratio)*100:.0f}%) salvato in {val_output}")
        self.log_to_file(f"- Test set: {len(test_df)} righe ({(1 - train_ratio - val_ratio)*100:.0f}%) salvato in {test_output}")
        
    def COCO_get_caption_and_category(self, img_name):
        # Trova l'immagine corrispondente
        for idx in self.captions_coco.imgs:
            if self.captions_coco.imgs[idx]["file_name"] == img_name:
                img_id = self.captions_coco.imgs[idx]["id"]
                ann_ids = self.captions_coco.getAnnIds(imgIds=img_id)
                captions = self.captions_coco.loadAnns(ann_ids)
                caption = captions[random.randint(0,len(captions)-1)]['caption']
                # Prendi le categorie dagli oggetti annotati (es. bounding box)
                obj_ann_ids = self.instances_coco.getAnnIds(imgIds=img_id)
                obj_anns = self.instances_coco.loadAnns(obj_ann_ids)
                category_ids = list(set([ann['category_id'] for ann in obj_anns]))

                # Se vuoi anche i nomi (non solo gli ID):
                #categories = [self.instances_coco.cats[cat_id]['name'] for cat_id in category_ids]
                
                if not category_ids:
                    supercategory = "empty"
                else:
                # Prendi le supercategorie associate
                    supercategories = list(set([self.instances_coco.cats[cat_id]['supercategory']
                                            for cat_id in category_ids]))
                    supercategory = random.choice(supercategories)
                return caption, supercategory
        
    def COCO_create_csv(self):        
        #with open(COCO_CSV_PATH,'w', newline='') as csv_file:
        with open("coco_with_category.csv",'w', newline='') as csv_file:
            fieldnames = ['image_path', 'caption','category']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for file_name in os.listdir(COCO_DATASET_PATH):
                caption, category = self.COCO_get_caption_and_category(file_name)
                writer.writerow({'image_path': COCO_DATASET_PATH+file_name, 'caption': caption, 'category': category})
                
    def merge_and_shuffle_csv(self, file1, file2, output_file, shuffle=True):
        # Leggi entrambi i file CSV
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    
        # Unisci i dataframe
        merged_df = pd.concat([df1, df2], ignore_index=True)
    
        # Mescola le righe se richiesto
        if shuffle:
            merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
        # Salva il risultato
        merged_df.to_csv(output_file, index=False)
    
        self.log_to_file(f"Merge completato:")
        self.log_to_file(f"- Righe dal primo file: {len(df1)}")
        self.log_to_file(f"- Righe dal secondo file: {len(df2)}")
        self.log_to_file(f"- Righe totali nel file unito: {len(merged_df)}")
        self.log_to_file(f"- File salvato in: {output_file}")

        
dataset = Annotations(train_size=10000,val_size=1000, nTrainPos=8000,nTrainNeg=2000,nValPos=800,nValNeg=200)

#dataset = Annotations(train_size=100,val_size=10, nTrainPos=80,nTrainNeg=20,nValPos=8,nValNeg=2)



