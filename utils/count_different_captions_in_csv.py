import pandas as pd
import json
'''
This script is used to count how many captions in a given csv are equal, ex:
"The image shows a sea turtle swimming alone in the ocean.": 1,
    "A sea turtle swims in the ocean, without another sea turtle in the background.": 1,
    "A sea turtle is swimming in the ocean without another sea turtle nearby.": 1,
    "A sea turtle alone in deep green water.": 1,
    "A sea turtle (not araffe, which is not a known term) swims in the ocean alongside a fish.": 1,
    "A sea turtle is swimming in the clear water of a lake.": 1,
    "One turtle moves through the vast sea.": 1,
    "A turtle swims unaccompanied in the vast ocean.": 1,
    "Two turtles share the ocean waters, swimming in close proximity.": 1, 

'''
# Read the CSV data into a pandas DataFrame
df = pd.read_csv("/workspace/text-to-image-retrivial/utils/definitivo.csv")

# Calculate the frequency of each caption
caption_counts = df['caption'].value_counts()
print(caption_counts.sum())

# Convert the Series to a dictionary
caption_counts_dict = caption_counts.to_dict()

# Print the dictionary
# Salva il dizionario in un file JSON
output_json_file = "caption_counts.json"
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(caption_counts_dict, f, ensure_ascii=False, indent=4)
    
print(len(caption_counts_dict.keys()))