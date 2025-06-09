import pandas as pd

# Leggi il file CSV originale
df = pd.read_csv('cropped_marine_dataset.csv')

# Filtra il dataframe per ogni categoria
turtle_df = df[df['category'] == 'turtle']
sea_df = df[df['category'] == 'sea']
debris_df = df[df['category'] == 'debris']
dolphin_df = df[df['category']=='dolphin']

print(len(turtle_df))
print(len(sea_df))
print(len(debris_df))
print(len(dolphin_df))


# Salva ciascun dataframe in un file CSV separato
turtle_df.to_csv('turtle_images.csv', index=False)
sea_df.to_csv('sea_images.csv', index=False)
debris_df.to_csv('debris_images.csv', index=False)
dolphin_df.to_csv('dolphin_images.csv', index=False)

print("File creati con successo!")