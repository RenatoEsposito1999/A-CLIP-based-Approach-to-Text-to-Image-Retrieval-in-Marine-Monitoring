import json
import random
import csv
from create_caption import generate_negative_sentence, generate_positive_sentence
import time
#from COCO_filter import get_caption
#from custom_utils.llm import LLM
import pandas as pd
from pycocotools.coco import COCO
import os
DATASET_ANNOTATIONS_PATH = "/projects/data/turtle-full-v2/annotations/instances_Train.json"
COCO_CSV_PATH = "COCO_all.csv"
CROPPED_TURTLE_CSV_PATH = "all_cropped_turtle_positive.csv"
COCO_DATASET_PATH="/projects/data/turtle-full-v2/COCO/"
CROPPED_TURTLE_POSITIVE_CSV_PATH = "all_cropped_turtle_positive.csv"
DATASET_IMAGES_CROPPED_PATH = "/projects/data/turtle-full-v2/images/Train_cropped/"
CAPTIONS_ANNOTATIONS_COCO_PATH = "/projects/data/turtle-full-v2/annotations/captions_val2014.json"

#POSSIBILI MODIFICHE DA FARE --> A COCO METTERE LE IMMAGINI CON LE KEYWORD ATTINENTI e/o usare COCO TRAIN

class Annotations:
    img_list = []
    def __init__(self, train_size,val_size,test_size,nTrainPos,nTrainNeg,nValPos,nValNeg,nTestPos,nTestNeg):
        self.img_list = self.extract_img()
        self.captions_coco = COCO(CAPTIONS_ANNOTATIONS_COCO_PATH)        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.nTrainPos = nTrainPos
        self.nTrainNeg = nTrainNeg
        self.nValPos = nValPos
        self.nValNeg = nValNeg
        self.nTestPos = nTestPos
        self.nTestNeg = nTestNeg
        self.LLM = None #self.LLM = LLM()
        
        #CREATE POSITIVE TURTLE CSV
        #self.turtle_create_csv()
        
        #CREATE NEGATIVE COCO CSV
        self.COCO_create_csv()

        #CREATE TRAINING, VALIDATION AND TEST SET
        self.split_csv(file1=CROPPED_TURTLE_CSV_PATH,file2=COCO_CSV_PATH)

    def split_csv(self,file1, file2, output_prefix='', random_state=None):
        """
        Divide due file CSV in training (80%), validation (10%) e test (10%) set.
        Args:
            file1 (str): Percorso del primo file CSV (fornirà 8000 train, 1000 val, 1000 test)
            file2 (str): Percorso del secondo file CSV (fornirà 16000 train, 2000 val, 2000 test)
            output_prefix (str): Prefisso per i file di output
            random_state (int): Seed per la riproducibilità
        """
        # Carica i file CSV
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Mescola i dataframe
        df1 = df1.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df2 = df2.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Verifica che ci siano abbastanza righe
        if len(df1) < (self.nTrainPos + self.nValPos + self.nTestPos): 
            raise ValueError(f"{file1} ha meno di 10000 righe (richieste: 8000 train + 1000 val + 1000 test)")
        
        if len(df2) < (self.nTrainNeg + self.nValNeg + self.nTestNeg): 
            raise ValueError(f"{file2} ha meno di 20000 righe (richieste: 16000 train + 2000 val + 2000 test)")
        
        # Split per il primo file (8000, 1000, 1000)
        train1 = df1.iloc[:8000]
        val1 = df1.iloc[8000:9000]
        test1 = df1.iloc[9000:10000]
        
        # Split per il secondo file (16000, 2000, 2000)
        train2 = df2.iloc[:16000]
        val2 = df2.iloc[16000:18000]
        test2 = df2.iloc[18000:20000]
        
        # Combina i dataset
        train = pd.concat([train1, train2], ignore_index=True)
        val = pd.concat([val1, val2], ignore_index=True)
        test = pd.concat([test1, test2], ignore_index=True)
        
        # Mescola i dataset combinati (mantenendo la proporzione ma mischiando le fonti)
        train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Salva i file
        train.to_csv(f"{output_prefix}training.csv", index=False)
        val.to_csv(f"{output_prefix}val.csv", index=False)
        test.to_csv(f"{output_prefix}test.csv", index=False)
        
        print("Split completato con successo!")
        print(f"Training set: {len(train)} righe (8000 da {file1} + 16000 da {file2})")
        print(f"Validation set: {len(val)} righe (1000 da {file1} + 2000 da {file2})")
        print(f"Test set: {len(test)} righe (1000 da {file1} + 2000 da {file2})")

    # Esempio di utilizzo:
    # split_csv('file1.csv', 'file2.csv', random_state=42)

    
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
    def COCO_get_caption(self, img_name):
        for idx in self.captions_coco.imgs:
            if self.captions_coco.imgs[idx]["file_name"] == img_name:
                img_id = self.captions_coco.imgs[idx]["id"]
                ann_ids = self.captions_coco.getAnnIds(imgIds=img_id)
                captions = self.captions_coco.loadAnns(ann_ids)
                return captions[random.randint(0,len(captions)-1)]['caption']
        
        
    def COCO_create_csv(self):        
        with open(COCO_CSV_PATH,'w', newline='') as csv_file:
            fieldnames = ['image_path', 'caption']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for file_name in os.listdir(COCO_DATASET_PATH):
                writer.writerow({'image_path': COCO_DATASET_PATH+file_name, 'caption': self.COCO_get_caption(file_name)})
                
def build_val_only_turtle(file1, file2, n_rows):
        df1 = pd.read_csv(file1)
        df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

        if not os.path.exists(file2):
            df2 = pd.DataFrame(columns=df1.columns)
        else:
            df2 = pd.read_csv(file2)
        
        rows = df1.head(n_rows)
        
        df2 = pd.concat([df2, rows], ignore_index=True)
        
        df2.to_csv(file2, index=False)
#build_val_only_turtle(file1="all_cropped_turtle_positive.csv", file2="only_turtle_val.csv", n_rows=1000)
dataset = Annotations(train_size=24000,val_size=3000, test_size = 3000, nTrainPos=8000,nTrainNeg=16000,nValPos=1000,nValNeg=2000,nTestPos=1000,nTestNeg=2000)

'''
Split	Tartarughe	Distrattori (COCO)	Totale
Train	8000	16000	24000
Validation	1000	2000	3000
Test	1000	2000	3000


'''