import pandas as pd
import json
# Read the CSV data into a pandas DataFrame
df = pd.read_csv("./image_captions.csv")

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