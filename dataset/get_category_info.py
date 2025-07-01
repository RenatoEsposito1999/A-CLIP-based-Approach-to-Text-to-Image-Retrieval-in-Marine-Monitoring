import os
import json
import csv
CATEGORY = {}
ID = 0
def build_category_json():
        with open("category_info.json", "w") as file_json:
            json.dump(CATEGORY,file_json, indent=2)

def store_category_info(category):
    global ID
    if category in CATEGORY:
        count = CATEGORY[category][1]
        CATEGORY[category][1] = count+1
    else:
        CATEGORY[category] = [ID,1]
        ID +=1


def process_file(dir_path):
    """Legge un file e estrae i campi"""
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            print(f"Processo file: {filename}")
                
            with open(file_path, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file,quotechar='"')
                next(csv_reader)
                for fields in csv_reader:
                        if len(fields) >= 4:
                            category = fields[3].strip().lower()
                            store_category_info(category)
    build_category_json()
 

