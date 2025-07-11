import os
def get_next_version(log_dir: str) -> int:
    """Trova il prossimo numero di versione disponibile."""
    os.makedirs(log_dir, exist_ok=True)  # Crea la cartella se non esiste
    existing_versions = []
    
    for d in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("version_"):
            try:
                num = int(d.split("_")[1])
                existing_versions.append(num)
            except ValueError:
                continue  # Ignora cartelle con formato non valido
    
    return max(existing_versions) + 1 if existing_versions else 0
