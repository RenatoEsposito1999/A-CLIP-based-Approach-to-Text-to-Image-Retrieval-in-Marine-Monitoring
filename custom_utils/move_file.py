import os
import random
import shutil

def move_random_files(source_dir, dest_dir, n, extensions=None, seed=None):
    """
    Sposta n file casuali da source_dir a dest_dir
    
    Args:
        source_dir (str): Percorso directory sorgente
        dest_dir (str): Percorso directory destinazione
        n (int): Numero di file da spostare
        extensions (list, optional): Estensioni da considerare (es. ['.jpg', '.png']). 
                                    Se None, considera tutti i file.
        seed (int, optional): Seed per la riproducibilità
    """
    # Crea la directory di destinazione se non esiste
    os.makedirs(dest_dir, exist_ok=True)
    
    # Imposta il seed per riproducibilità
    if seed is not None:
        random.seed(seed)
    
    # Ottieni lista file con filtri
    all_files = []
    for f in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, f)):
            if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                all_files.append(f)
    
    if not all_files:
        print("Nessun file trovato nella directory sorgente")
        return
    
    # Verifica che n non sia maggiore dei file disponibili
    n = min(n, len(all_files))
    
    # Seleziona file casuali
    selected_files = random.sample(all_files, n)
    
    # Sposta i file
    for file in selected_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(dest_dir, file)
        
        shutil.move(src_path, dst_path)
        print(f"Spostato: {file}")
    
    print(f"\nSpostati {n} file da {source_dir} a {dest_dir}")

# Esempio di utilizzo
if __name__ == "__main__":
    input_dir = "/workspace/text-to-image-retrivial/NEW_DATASET/Train/COCO-FLICKR30"       # Sostituisci con la tua directory di input
    output_dir = "/workspace/text-to-image-retrivial/NEW_DATASET/Test/COCO-Flickr30k"        # Sostituisci con la tua directory di output
    num_files = 800                  # Numero di file da spostare
    file_extensions = ['.jpg', '.png']  # Estensioni da considerare (None per tutte)
    
    move_random_files(input_dir, output_dir, num_files, extensions=file_extensions, seed=42)