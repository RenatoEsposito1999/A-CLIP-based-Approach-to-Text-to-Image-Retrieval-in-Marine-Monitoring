import kagglehub

# Download latest version
path = kagglehub.dataset_download("wildlifedatasets/seaturtleid2022")

print("Path to dataset files:", path)