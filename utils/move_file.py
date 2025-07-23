import os
import random
import shutil

def move_random_files(source_dir, dest_dir, n, extensions=None, seed=None):
    """
    Move n random files from source_dir to dest_dir
    
    Args:
        source_dir (str): Path to source directory
        dest_dir (str): Path to destination directory  
        n (int): Number of files to move
        extensions (list, optional): File extensions to consider (e.g. ['.jpg', '.png']).
                                    If None, all files are considered.
        seed (int, optional): Random seed for reproducibility
    """
    #Create directory if does not exist
    os.makedirs(dest_dir, exist_ok=True)
    
    #Set the seed
    if seed is not None:
        random.seed(seed)
    
    #Obtain the list with all path of images
    all_files = []
    for f in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, f)):
            if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                all_files.append(f)
    
    if not all_files:
        print("No files found in the source directory")
        return
    
    #Check if the number of files that want to move is not greater than the effective number of images inside the directory
    n = min(n, len(all_files))
    
    #Select random files
    selected_files = random.sample(all_files, n)
    
    #Move the files
    for file in selected_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(dest_dir, file)
        
        shutil.move(src_path, dst_path)
        print(f"Moved: {file}")
    
    print(f"\Moved {n} files from {source_dir} to {dest_dir}")

def main():
    input_dir = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/COCO-Flickr30k"      
    output_dir = "/workspace/text-to-image-retrivial/NEW_DATASET/Validation/COCO-Flickr30k"     
    num_files = 13000                  
    file_extensions = ['.jpg', '.png']  
    move_random_files(input_dir, output_dir, num_files, extensions=file_extensions, seed=42)
    
if __name__ == "__main__":
    main()