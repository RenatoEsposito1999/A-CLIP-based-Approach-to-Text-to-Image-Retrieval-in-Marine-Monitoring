import random
import csv
from create_caption import generate_positive_sentence
import time
import os

CROPPED_TURTLE_POSITIVE_CSV_PATH = "./Turtle.csv"
DATASET_IMAGES_CROPPED_PATH = "/workspace/text-to-image-retrivial/datasets/Turtle"

def create_turtle_csv(LLM=None):
    turtle_file = open(CROPPED_TURTLE_POSITIVE_CSV_PATH, mode='w', encoding='utf-8', newline='')
    fieldnames = ['image_name','caption']
    writer_turtle = csv.DictWriter(turtle_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer_turtle.writeheader()
    
    for file_name in os.listdir(DATASET_IMAGES_CROPPED_PATH):
        dynamic_random = random.Random(time.time())
        mantain_templating = dynamic_random.random() < 0.3 # 30% chance of not using the llm
        sentence = generate_positive_sentence()
        if not mantain_templating and LLM:
            sentence = LLM.rephrase_sentence(sentence=sentence)
                
        writer_turtle.writerow({"image_name": f"cropped_{file_name}", "caption": sentence})
    turtle_file.close()
    
def main():
    create_turtle_csv()

if __name__ == '__main__':
    main()
