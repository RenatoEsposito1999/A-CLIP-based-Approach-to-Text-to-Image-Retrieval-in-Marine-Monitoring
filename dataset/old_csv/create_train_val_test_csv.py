import pandas as pd

# Load and shuffle the datasets
df_turtle = pd.read_csv('turtle_images.csv').sample(frac=1, random_state=42)
df_debris = pd.read_csv('debris_images.csv').sample(frac=1, random_state=42)
df_dolphin = pd.read_csv('dolphin_images.csv').sample(frac=1, random_state=42)
df_coco = pd.read_csv('COCO_with_category.csv').sample(frac=1, random_state=42)

# Define train sizes (as per your request)
train_turtle = 12000
train_debris = 544
train_dolphin = 484
train_COCO = 20000

# Calculate validation sizes (half of remaining)
val_turtle = int(((len(df_turtle))-train_turtle) / 2)
val_debris = int(((len(df_debris))-train_debris) / 2)
val_dolphin = int(((len(df_dolphin))-train_dolphin) / 2)
val_COCO = int(((len(df_coco))-train_COCO) / 2)

# Calculate test sizes (the rest)
test_turtle = (len(df_turtle) - train_turtle - val_turtle)
test_debris = (len(df_debris) - train_debris - val_debris)
test_dolphin = (len(df_dolphin) - train_dolphin - val_dolphin)
test_COCO = (len(df_coco) - train_COCO - val_COCO)

# Split each dataset into train, val, test
def split_dataframe(df, train_size, val_size, test_size):
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size+val_size]
    test = df.iloc[train_size+val_size:train_size+val_size+test_size]
    return train, val, test

# Turtle splits
turtle_train, turtle_val, turtle_test = split_dataframe(
    df_turtle, train_turtle, val_turtle, test_turtle
)

# Debris splits
debris_train, debris_val, debris_test = split_dataframe(
    df_debris, train_debris, val_debris, test_debris
)

# Dolphin splits
dolphin_train, dolphin_val, dolphin_test = split_dataframe(
    df_dolphin, train_dolphin, val_dolphin, test_dolphin
)

# COCO splits
coco_train, coco_val, coco_test = split_dataframe(
    df_coco, train_COCO, val_COCO, test_COCO
)

# Combine all datasets
train_df = pd.concat([turtle_train, debris_train, dolphin_train, coco_train], axis=0).sample(frac=1, random_state=42)
val_df = pd.concat([turtle_val, debris_val, dolphin_val, coco_val], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([turtle_test, debris_test, dolphin_test, coco_test], axis=0).sample(frac=1, random_state=42)

# Save to CSV files
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('val_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("Datasets created successfully!")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")


