import os
from PIL import Image

def get_image_dimensions_in_folder(folder_path):
    """
    Calcola e stampa le dimensioni (larghezza x altezza) di tutte le immagini 
    presenti nella cartella specificata.

    Args:
        folder_path (str): Il percorso della cartella da scansionare.

    Returns:
        dict: Un dizionario che mappa i nomi dei file alle loro dimensioni.
    """
    image_sizes = {}
    
    # Estensioni comuni dei file immagine
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif')
    
    print(f"Scansione della cartella: {folder_path}...")
    
    # Itera su tutti i file nella cartella
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 1. Verifica che sia un file (non una sottocartella)
        if os.path.isfile(file_path):
            
            # 2. Verifica che l'estensione sia di un'immagine
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                try:
                    # Apri l'immagine con Pillow (PIL)
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        # Memorizza il risultato
                        image_sizes[filename] = f"{width}x{height}"
                        print(f"✅ {filename}: {width}x{height}")
                        
                except Exception as e:
                    # Gestisce gli errori (es. file corrotto o formato non supportato)
                    image_sizes[filename] = "ERRORE nel caricamento"
                    print(f"❌ {filename}: ERRORE nel caricamento - {e}")
            
            # 3. Ignora gli altri tipi di file
            else:
                # print(f"-> Ignorato: {filename} (non è un'immagine supportata)")
                pass
    
    # Stampa un riepilogo
    print("\n--- Riepilogo Dimensioni Immagini ---")
    for name, size in image_sizes.items():
        print(f"{name}: {size}")
        
    return image_sizes

# =========================================================================
# IMPOSTAZIONI: MODIFICA QUESTO VALORE
# =========================================================================
# Inserisci il percorso della cartella contenente le tue immagini
FOLDER_TO_CHECK = "SR\ARSGN\data\hr"

# Esecuzione della funzione
if __name__ == "__main__":
    if not os.path.isdir(FOLDER_TO_CHECK):
        print(f"ERRORE: La cartella specificata non esiste: {FOLDER_TO_CHECK}")
    else:
        results = get_image_dimensions_in_folder(FOLDER_TO_CHECK)
