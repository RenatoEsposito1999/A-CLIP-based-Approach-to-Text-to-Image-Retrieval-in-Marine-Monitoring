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

COCO_TXT_PATH = "COCO_with_category.txt"
CROPPED_TURTLE_POSITIVE_TXT_PATH = "cropped_turtle.txt"
CROPPED_DEBRIS_TXT_PATH = "cropped_debris.txt"
CROPPED_DOLPHINE_TXT_PATH = "cropped_dolphine.txt"
CROPPED_SEA_TXT_PATH = "cropped_sea.txt"

DATASET_ANNOTATIONS_PATH = "/workspace/annotations/instances_Train.json"
COCO_DATASET_PATH = "/workspace/text-to-image-retrivial/datasets/images/COCO/"
DATASET_IMAGES_CROPPED_PATH = "/workspace/text-to-image-retrivial/datasets/images/Train_cropped/"
CAPTIONS_ANNOTATIONS_COCO_PATH = "/workspace/annotations/captions_val2014.json"
COCO_ISTANCES_VAL_PATH = "/workspace/annotations/instances_val2014.json"
OTHER_TURTLE_DATASET = "/workspace/text-to-image-retrivial/datasets/images/turtle/turtles-data/data/images"

'''DATASET_ANNOTATIONS_PATH = "/projects/data/turtle-full-v2/annotations/instances_Train.json"
COCO_DATASET_PATH="/projects/data/turtle-full-v2/COCO/"
DATASET_IMAGES_CROPPED_PATH = "/projects/data/turtle-full-v2/images/Train_cropped/"
CAPTIONS_ANNOTATIONS_COCO_PATH = "/projects/data/turtle-full-v2/annotations/captions_val2014.json"
COCO_ISTANCES_VAL_PATH = "/projects/data/turtle-full-v2/annotations/instances_val2014.json"'''


'''DATASET_ANNOTATIONS_PATH = "/Volumes/Seagate/annotations/instances_Train.json"
COCO_DATASET_PATH="/Volumes/Seagate/COCO/"
DATASET_IMAGES_CROPPED_PATH = "/Volumes/Seagate/images/Train_cropped/"
CAPTIONS_ANNOTATIONS_COCO_PATH = "/Volumes/Seagate/annotations/captions_val2014.json"
COCO_ISTANCES_VAL_PATH = "/Volumes/Seagate/annotations/instances_val2014.json"
'''
#POSSIBILI MODIFICHE DA FARE --> A COCO METTERE LE IMMAGINI CON LE KEYWORD ATTINENTI e/o usare COCO TRAIN

class Annotations:
    img_list = []
    ID = 0
    def __init__(self, train_size,val_size,test_size,nTrainPos,nTrainNeg,nValPos,nValNeg,nTestPos,nTestNeg):
        self.img_list = self.extract_img()
        self.captions_coco = COCO(CAPTIONS_ANNOTATIONS_COCO_PATH)        
        self.instances_coco = COCO(COCO_ISTANCES_VAL_PATH)    
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.nTrainPos = nTrainPos
        self.nTrainNeg = nTrainNeg
        self.nValPos = nValPos
        self.nValNeg = nValNeg
        self.nTestPos = nTestPos
        self.nTestNeg = nTestNeg
        self.category = {}
        self.LLM = None #self.LLM = LLM()
        
        #CREATE POSITIVE TURTLE CSV
        self.turtle_create_txt()
        
        #CREATE NEGATIVE COCO CSV
        self.COCO_create_txt()
        #CREATE TRAINING, VALIDATION AND TEST SET
        self.split_2_txt(file1=CROPPED_TURTLE_POSITIVE_TXT_PATH,file2=COCO_TXT_PATH)
        #self.split_1_csv(file1=CROPPED_TURTLE_POSITIVE_CSV_PATH)
        #self.category_info()

    def build_category_json(self):
        with open("category_info.json", "w") as file_json:
            json.dump(self.category,file_json, indent=2)


    '''def split_1_csv(self,file1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=None):
        """
        Split a CSV file into train, validation, and test sets (default: 80/10/10).
        
        Args:
            file1 (str): Path to the input CSV file.
            train_ratio (float): Proportion for training set (default: 0.8).
            val_ratio (float): Proportion for validation set (default: 0.1).
            test_ratio (float): Proportion for test set (default: 0.1).
            random_state (int): Seed for reproducibility (optional).
        
        Returns:
            None (saves train.csv, val.csv, test.csv in the same directory).
        """
        # Read the input CSV
        df = pd.read_csv(file1)
        
        # Shuffle the dataset (optional, but recommended)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate split indices
        train_end = int(len(df) * train_ratio)
        val_end = train_end + int(len(df) * val_ratio)
        
        # Split into train, val, test
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        # Save to CSV (same directory as input file)
        output_dir = "/".join(file1.split("/")[:-1]) if "/" in file1 else "."
        train_df.to_csv(f"{output_dir}/training.csv", index=False)
        val_df.to_csv(f"{output_dir}/val.csv", index=False)
        test_df.to_csv(f"{output_dir}/test.csv", index=False)
        
        print(f"Split completed: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")'''

    def split_2_txt(self, file1, file2, output_prefix='', random_state=None):
        """
        Divide due file TXT in training (80%), validation (10%) e test (10%) set.
        Formato input/output: 'image_name, comment_number, comment'
        Args:
            file1 (str): Percorso del primo file TXT (fornirà 8000 train, 1000 val, 1000 test)
            file2 (str): Percorso del secondo file TXT (fornirà 16000 train, 2000 val, 2000 test)
            output_prefix (str): Prefisso per i file di output
            random_state (int): Seed per la riproducibilità
        """
        # Funzione per caricare un file TXT in un DataFrame
        def load_txt_to_df(txt_path):
            with open(txt_path, 'r') as f:
                lines = [line.strip().split(', ', 2) for line in f.readlines() if line.strip()]
            return pd.DataFrame(lines, columns=['image_name', 'comment_number', 'comment'])

        # Carica i file TXT
        df1 = load_txt_to_df(file1)
        df2 = load_txt_to_df(file2)
        
        # Mescola i dataframe
        df1 = df1.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df2 = df2.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Verifica che ci siano abbastanza righe (stessi controlli originali)
        if len(df1) < (self.nTrainPos + self.nValPos + self.nTestPos): 
            raise ValueError(f"{file1} ha meno di 10000 righe (richieste: 8000 train + 1000 val + 1000 test)")
        if len(df2) < (self.nTrainNeg + self.nValNeg + self.nTestNeg): 
            raise ValueError(f"{file2} ha meno di 20000 righe (richieste: 16000 train + 2000 val + 2000 test)")
        
        # Split per il primo file (8000 train, 1000 val, 1000 test) - TURTLE
        train1 = df1.iloc[2000:10000]  # 8k
        val1 = df1.iloc[11500:12500]   # 1k
        test1 = df1.iloc[14000:15000]  # 1k
        
        # Split per il secondo file (16000 train, 2000 val, 2000 test) - COCO
        train2 = df2.iloc[:12000]       # 12k (originale: 16k, ma hai solo 12k righe nel tuo esempio)
        val2 = df2.iloc[12000:14000]   # 2k
        test2 = df2.iloc[14000:16000]  # 2k
        
        # Combina i dataset
        train = pd.concat([train1, train2], ignore_index=True)
        val = pd.concat([val1, val2], ignore_index=True)
        test = pd.concat([test1, test2], ignore_index=True)
        
        # Mescola i dataset combinati
        train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Funzione per salvare DataFrame come TXT
        def save_df_to_txt(df, output_path):
            with open(output_path, 'w') as f:
                f.write("image_name, comment_number, comment\n")  # Header
                for _, row in df.iterrows():
                    f.write(f"{row['image_name']}, {row['comment_number']}, {row['comment']}\n")
        
        # Salva i file TXT
        save_df_to_txt(train, f"{output_prefix}training.txt")
        save_df_to_txt(val, f"{output_prefix}val.txt")
        save_df_to_txt(test, f"{output_prefix}test.txt")
        
        print("Split completato con successo!")
        print(f"Training set: {len(train)} righe ({len(train1)} da {file1} + {len(train2)} da {file2})")
        print(f"Validation set: {len(val)} righe ({len(val1)} da {file1} + {len(val2)} da {file2})")
        print(f"Test set: {len(test)} righe ({len(test1)} da {file1} + {len(test2)} da {file2})")

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
        idx = 0
        for img in self.img_list:
            if img[0] == "frame_054649.PNG":
                idx = idx + 1
        return self.img_list

    def turtle_create_txt(self):
        turtle_txt = open(CROPPED_TURTLE_POSITIVE_TXT_PATH,'w', newline='')
        debris_txt = open(CROPPED_DEBRIS_TXT_PATH,'w', newline='')
        dolphine_txt = open(CROPPED_DOLPHINE_TXT_PATH,'w', newline='')
        sea_txt = open(CROPPED_SEA_TXT_PATH,'w', newline='')
        turtle_txt.write("image_name, comment_number, comment\n")
        debris_txt.write("image_name, comment_number, comment\n")
        dolphine_txt.write("image_name, comment_number, comment\n")
        sea_txt.write("image_name, comment_number, comment\n")
        for img_name in self.img_list:
            idx = 0
            if img_name[1] == 1: #img with dolphine
                dynamic_random = random.Random(time.time())
                mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                sentence = generate_dolphine_sentence()
                if not mantain_templating and self.LLM:
                    sentence = self.LLM.rephrase_sentence(sentence=sentence)
                for idx in range(5):
                    dolphine_txt.write(f"cropped_{img_name[0]}, {idx}, {sentence}\n")
                
                #self.store_category_info("dolphin")
            if img_name[1] == 2: #img with turt == 2
                dynamic_random = random.Random(time.time())
                mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                sentence = generate_positive_sentence()
                if not mantain_templating and self.LLM:
                    sentence = self.LLM.rephrase_sentence(sentence=sentence)
                for idx in range(5):
                    turtle_txt.write(f"cropped_{img_name[0]}, {idx}, {sentence}\n")
                
                #self.store_category_info("turtle")
            if img_name[1] == 3: ## img with trash
                dynamic_random = random.Random(time.time())
                mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                sentence = generate_negative_sentence(include_trash=True)
                if not mantain_templating and self.LLM:
                    sentence = self.LLM.rephrase_sentence(sentence=sentence)
                for idx in range(5):
                    debris_txt.write(f"cropped_{img_name[0]}, {idx}, {sentence}\n")
                    
                #self.store_category_info("debris")
            elif img_name[1] == 4: #other area sea view
                dynamic_random = random.Random(time.time())
                mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
                sentence = generate_negative_sentence(include_trash=False)
                if not mantain_templating and self.LLM:
                    sentence = self.LLM.rephrase_sentence(sentence=sentence)
                for idx in range(5):
                    sea_txt.write(f"cropped_{img_name[0]}, {idx}, {sentence}\n")
                #self.store_category_info("sea")
        debris_txt.close()
        sea_txt.close()
        turtle_txt.close()
        dolphine_txt.close()

    def store_category_info(self,category):
        if category in self.category:
            count = self.category[category][1]
            self.category[category][1] = count+1
        else:
            self.category[category] = [self.ID,1]
            self.ID +=1

    def COCO_get_caption_and_category(self, img_name):
        #Trova l'immagine corrispondente
        for idx in self.captions_coco.imgs:
            if self.captions_coco.imgs[idx]["file_name"] == img_name:
                img_id = self.captions_coco.imgs[idx]["id"]
                ann_ids = self.captions_coco.getAnnIds(imgIds=img_id)
                captions = self.captions_coco.loadAnns(ann_ids)
                only_captions = []
                for caption in captions:
                    only_captions.append(caption["caption"])
                #caption = captions[random.randint(0,len(captions)-1)]['caption']
                #obj_ann_ids = self.instances_coco.getAnnIds(imgIds=img_id)
                #obj_anns = self.instances_coco.loadAnns(obj_ann_ids)
                #category_ids = list(set([ann['category_id'] for ann in obj_anns]))
                '''if not category_ids:
                    supercategory = "empty"
                else:
                    supercategories = list(set([self.instances_coco.cats[cat_id]['supercategory']
                                            for cat_id in category_ids]))
                    supercategory = random.choice(supercategories)'''
                
                return only_captions
        
    def COCO_create_txt(self):        
        with open(COCO_TXT_PATH, 'w') as txt_file:  # Apri il file in modalità scrittura
            txt_file.write("image_name, comment_number, comment\n")
            for file_name in os.listdir(COCO_DATASET_PATH):
                # Ottieni la lista di caption e la categoria
                captions = self.COCO_get_caption_and_category(file_name)
                       
                # Per ogni caption nella lista, scrivi una riga nel file .txt
                for i, caption in enumerate(captions):
                    # Formato: image_name, comment_number, comment
                    caption = caption.replace("\n", "")
                    txt_file.write(f"{file_name}, {i}, {caption}\n")
    
    def category_info(self):
        with open("training.csv", mode='r', encoding='utf-8') as file:
        #with open("cropped_marine_dataset.csv", mode='r', encoding='utf-8') as file:
            # Utilizza DictReader per accedere ai campi per nome
            reader = csv.DictReader(file)
            for row in reader:
                if 'category' in row:
                    self.store_category_info(row['category'])
        self.build_category_json()



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
    
        
#build_val_only_turtle(file1="cropped_marine_dataset.csv", file2="only_turtle_val.csv", n_rows=5000)
dataset = Annotations(train_size=24000,val_size=3000, test_size = 3000, nTrainPos=8000,nTrainNeg=16000,nValPos=1000,nValNeg=2000,nTestPos=1000,nTestNeg=2000)

'''
Split	Tartarughe	Distrattori (COCO)	Totale
Train	8000	16000	24000
Validation	1000	2000	3000
Test	1000	2000	3000
'''