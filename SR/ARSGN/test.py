import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Importa i moduli del tuo progetto
import config
from models.agsr import Net
from utils.dataset import SRDataset, get_dataloaders # Usiamo SRDataset per il test
# Nota: La classe SRDataset è contenuta in utils/dataset.py



def save_image(tensor, filename):
    """
    Salva un tensore PyTorch come file immagine (PNG).
    """
    # Sposta il tensore su CPU e rimuovi la dimensione Batch (se presente)
    tensor = tensor.cpu().squeeze(0)
    
    # Riconverte i valori da [0.0, 1.0] a [0, 255]
    # E riordina le dimensioni da (C, H, W) a (H, W, C) per PIL
    numpy_image = tensor.mul(255).clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    
    # Crea e salva l'immagine PIL
    image = Image.fromarray(numpy_image)
    image.save(filename)


def test():
    print(f"--- Avvio Inferenza su {config.DEVICE} ---")
    
    # 1. INIZIALIZZAZIONE
    # ----------------------------------------------------------------------
    device = torch.device(config.DEVICE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Simula l'oggetto args
    class Args: pass 
    args = Args()
    
    # 2. CARICAMENTO MODELLO
    # ----------------------------------------------------------------------
    model = Net(args).to(device)
    
    # Definisci il percorso del modello da caricare
    # DEVI CAMBIARE QUESTA VARIABILE per puntare al tuo modello addestrato
    WEIGHTS_PATH = os.path.join(config.WEIGHTS_DIR, 'latest_model.pth') 
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERRORE: Pesi del modello non trovati in {WEIGHTS_PATH}")
        return
        
    print(f"Caricamento pesi da: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval() # Imposta il modello in modalità valutazione/inferenza
    
    # 3. CARICAMENTO DATI DI TEST
    # ----------------------------------------------------------------------
    # Utilizziamo la stessa logica di dataset, ma non abbiamo bisogno di shuffle
    test_loader = get_dataloaders(
        lr_dir=config.DATA_DIR_LR, 
        hr_dir=config.DATA_DIR_HR, # Si usa hr_dir solo per ottenere il numero di file
        batch_size=1, # Inferenza sempre con batch_size=1
        shuffle=False
    )
    
    # 4. CICLO DI INFERENZA
    # ----------------------------------------------------------------------
    total_images = len(test_loader.dataset)
    
    with torch.no_grad(): # Disabilita il calcolo del gradiente per risparmiare memoria
        for idx, (lr_batch, hr_batch) in enumerate(test_loader):
            lr_batch = lr_batch.to(device) 
            
            # Forward Pass
            # Il modello restituisce (sr_1, sr_2). sr_2 è l'output finale.
            _, sr_final = model(lr_batch)
            
            # 5. SALVATAGGIO DEI RISULTATI
            # --------------------------------------------------------------
            
            # Ottieni il nome del file originale per il salvataggio
            # Nota: questo richiede che l'indice del dataset corrisponda ai percorsi originali
            original_filename = os.path.basename(test_loader.dataset.hr_paths[idx])
            output_filename = os.path.join(OUTPUT_DIR, f"SR_{original_filename}")
            
            save_image(sr_final, output_filename)
            
            print(f"Elaborato e salvato: {output_filename} ({idx + 1}/{total_images})")

    print("--- Inferenza Completata ---")


if __name__ == '__main__':
    test()