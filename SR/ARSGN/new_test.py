import os
import torch
import torch.nn as nn
import numpy as np
# Rimosse astropy.io e le sue funzioni (fits)
import sys
import torch.nn.functional as F

# Nuove importazioni necessarie per TIFF
from skimage.io import imread, imsave 
from skimage.util import img_as_float32 # Utile per standardizzare il caricamento
# Importa la tua funzione di downsampling
from utils.preprocessing import load_and_downsample_tif

# Importa i moduli del tuo progetto
import config
from models.agsr import Net


# --- NUOVE FUNZIONI DI CARICAMENTO E SALVATAGGIO TIFF ---

def load_tif_data(filepath):
    """
    Carica un file TIFF e lo converte in un tensore PyTorch (1, C, H, W).
    Normalizza i dati a float32 e, se sono già a float, li assume normalizzati.
    """
    try:
        # Carica i dati e convertili in float32 standardizzato
        # img_as_float32 scala i dati interi (es. uint16) nell'intervallo [0.0, 1.0]
        # Se il tuo TIFF è già scientifico non normalizzato, potresti dover 
        # rimuovere questa riga e usare solo .astype(np.float32) + normalizzazione manuale.
        # Mantengo img_as_float32 come standard per le immagini.
        img_np = img_as_float32(imread(filepath))
        
        if img_np.ndim == 2:
            # Immagine in scala di grigi (H, W) -> (1, 1, H, W)
            # Aggiunge dimensione Canale (1) e dimensione Batch (1)
            img_tensor = torch.from_numpy(img_np[np.newaxis, np.newaxis, ...])
        elif img_np.ndim == 3:
            # Immagine a colori (H, W, C) -> (1, C, H, W)
            # Trasposta per C al posto di H e W, poi aggiunge dimensione Batch (1)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            
        print(f"Dati TIFF caricati: {img_np.shape}. Tensore PyTorch: {img_tensor.shape}")
        
        # Sposta al device prima dell'inferenza
        return img_tensor.to(config.DEVICE)
        
    except Exception as e:
        print(f"ERRORE nel caricamento TIFF di {filepath}: {e}")
        return None


def save_tif_image(data_tensor, output_filepath):
    """
    Salva un tensore PyTorch (1, C, H, W) o (1, 1, H, W) come file TIFF.
    
    Args:
        data_tensor (torch.Tensor): Il tensore dell'immagine SR.
        output_filepath (str): Il percorso dove salvare l'immagine.
    """
    try:
        # 1. Sposta su CPU e converte in array NumPy
        # Rimuove la dimensione Batch (0)
        img_np = data_tensor.squeeze(0).cpu().numpy()
        
        # 2. Gestione dei canali
        if img_np.ndim == 3:
            # Immagine a colori (C, H, W) -> (H, W, C) per il salvataggio standard
            img_np = img_np.transpose(1, 2, 0)
        elif img_np.ndim == 2:
            # Immagine in scala di grigi (H, W)
            pass
        
        # 3. Gestione della gamma dinamica (CRITICA PER I FILE .TIF)
        # Se il tuo modello SR produce output non normalizzati, devi scalare
        # qui. Se invece l'output è normalizzato [0.0, 1.0], puoi lasciarlo float.
        # ASSUMO che l'output SR sia normalizzato [0.0, 1.0]
        
        # Se vuoi salvarlo come 16-bit, de-normalizza:
        # max_val_16bit = 65535.0
        # img_np = (np.clip(img_np, 0.0, 1.0) * max_val_16bit).astype(np.uint16)
        
        # In questo caso, lo salviamo come float32 non normalizzato [0.0, 1.0] per preservare i valori
        img_np = img_np.astype(np.float32)
        
        imsave(output_filepath, img_np)
        
    except Exception as e:
        print(f"ERRORE nel salvataggio TIFF di {output_filepath}: {e}")



# --- FUNZIONI DI INFERENZA CON TILING ---

def pad_for_tiling(image_tensor, patch_size, device):
    """
    Applica il padding a un'immagine tensore (1, C, H, W) in modo che H e W siano 
    multipli esatti della patch_size.
    
    Restituisce il tensore padded e le dimensioni originali per il cropping finale.
    """
    _, C, H, W = image_tensor.shape
    
    # Calcola il padding necessario
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size
    
    # Applica il padding (destra e basso)
    # [left, right, top, bottom]
    # Usa 'replicate' o 'reflect' per ridurre gli artefatti, se appropriato
    padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='replicate')
    
    print(f"Padding aggiunto: ({pad_h} righe, {pad_w} colonne). Nuova forma: {padded_tensor.shape}")
    return padded_tensor, H, W

def inferenza_con_tiling(model, lr_full_tensor, scale_factor, patch_size, device):
    """
    Esegue l'inferenza patch-wise sull'intera immagine LR e ricostruisce l'immagine SR.
    

    Args:
        model (nn.Module): Il modello SR addestrato.
        lr_full_tensor (torch.Tensor): L'immagine LR completa (1, 1, H_lr, W_lr).
        scale_factor (int): Il fattore di ingrandimento (es. 4).
        patch_size (int): La dimensione del patch LR usata per l'addestramento (es. 64).
        device (torch.device): Il dispositivo di calcolo.

    Returns:
        torch.Tensor: L'immagine SR completa (1, 1, H_sr, W_sr).
    """
    
    # 1. Prepara l'immagine con padding
    lr_padded, H_orig, W_orig = pad_for_tiling(lr_full_tensor, patch_size, device)
    _, C, H_pad, W_pad = lr_padded.shape
    
    # Inizializza il tensore SR di output (grande quanto H_pad * scale, W_pad * scale)
    SR_H_pad = H_pad * scale_factor
    SR_W_pad = W_pad * scale_factor
    sr_reconstructed = torch.zeros((1, C, SR_H_pad, SR_W_pad), device=device)
    
    # 2. Ciclo sui Patch
    print(f"Avvio inferenza su {H_pad // patch_size}x{W_pad // patch_size} patch...")
    
    for i in range(0, H_pad, patch_size):
        for j in range(0, W_pad, patch_size):
            
            # 2a. Estrai il patch LR
            lr_patch = lr_padded[:, :, i:i + patch_size, j:j + patch_size].to(device)
            
            # 2b. Inferenza sul Patch
            # L'output è SR_patch di dimensione (scale*patch_size) x (scale*patch_size)
            # Assumiamo che il forward del modello restituisca direttamente l'output SR
            # Se il tuo modello restituisce (residual, sr_final) come nell'esempio, usa:
            ar_patch, sr_patch = model(lr_patch)
            
            # sr_patch = model(lr_patch)
            
            # 2c. Ricostruisci l'immagine SR
            SR_i = i * scale_factor
            SR_j = j * scale_factor
            SR_patch_size = patch_size * scale_factor
            
            sr_reconstructed[:, :, SR_i:SR_i + SR_patch_size, SR_j:SR_j + SR_patch_size] = sr_patch
            
            # Potresti voler implementare un overlap qui per evitare artefatti,
            # ma lo teniamo semplice per questa implementazione.

    # 3. Rimuovi il padding SR
    SR_H_orig = H_orig * scale_factor
    SR_W_orig = W_orig * scale_factor
    sr_final = sr_reconstructed[:, :, :SR_H_orig, :SR_W_orig]
    
    return sr_final.contiguous(), ar_patch


# --- FUNZIONE PRINCIPALE DI TEST (MODIFICATA) ---

def test():
    """Esegue l'inferenza sull'immagine LR completa utilizzando il Tiling."""
    print(f"--- Avvio Inferenza con Tiling su {config.DEVICE} per dati TIFF ---")
    
    # 1. INIZIALIZZAZIONE
    # ----------------------------------------------------------------------
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Simula l'oggetto args (necessario per inizializzare il modello Net(args))
    class Args: pass 
    # **NOTA:** Dovrai popolare `args` con i parametri reali del tuo modello
    # (es. args.scale, args.n_resblocks, ecc.)
    args = Args() 
    
    # Variabili di tiling
    PATCH_SIZE = config.HR_PATCH_SIZE // config.SCALE_FACTOR 
    # Ho corretto la logica: se HR_PATCH_SIZE è 256 e SCALE_FACTOR è 4, il patch LR è 64
    
    # 2. CARICAMENTO MODELLO
    # ----------------------------------------------------------------------
    model = Net(args).to(device)
    WEIGHTS_PATH = os.path.join(config.WEIGHTS_DIR, 'model_epoch_100.pth') 
    
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"ERRORE: Pesi del modello non trovati in {WEIGHTS_PATH}")
        
    print(f"Caricamento pesi da: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    # 3. CARICAMENTO ELENCO FILE DI TEST
    # ----------------------------------------------------------------------
    # Utilizza os.listdir per ottenere l'elenco dei file TIFF nella directory LR
    lr_files = [f for f in os.listdir(config.DATA_DIR_TEST_LR) if f.lower().endswith(('.tif', '.tiff'))]
    total_images = len(lr_files)
    
    # 4. CICLO DI INFERENZA IMMAGINE INTERA
    # ----------------------------------------------------------------------
    with torch.no_grad():
        for idx, filename in enumerate(lr_files):
            original_lr_path = os.path.join(config.DATA_DIR_TEST_LR, filename)
            
            print(f"\n--- Elaborazione {filename} ({idx + 1}/{total_images}) ---")
            
            # 4a. Carica l'immagine LR intera (RISULTATO: 1, C, H, W tensor)
            lr_full_tensor = load_tif_data(original_lr_path)
            
            if lr_full_tensor is None:
                continue 
            
            # 4b. Esegui l'inferenza con Tiling
            sr_final_tensor, ar_tensor = inferenza_con_tiling(
                model, 
                lr_full_tensor, 
                config.SCALE_FACTOR, 
                PATCH_SIZE, 
                device
            )
            
            # 4c. SALVATAGGIO DEI RISULTATI
            # --------------------------------------------------------------
            
            base_name, _ = os.path.splitext(filename)
            # Salva in formato TIFF
            output_filename = os.path.join(config.OUTPUT_DIR, f"SR_{base_name}_x{config.SCALE_FACTOR}.tif")
            output_ar_filename = os.path.join(config.OUTPUT_DIR, f"AR_{base_name}_x{config.SCALE_FACTOR}.tif")
            
            save_tif_image(sr_final_tensor, output_filename)
            save_tif_image(ar_tensor.squeeze(0), output_ar_filename) # AR ha bisogno di un squeeze se è solo un patch
            
            print(f"-> Salvato: {output_filename}", f"-> Salvato: {output_ar_filename}")

    print("--- Inferenza Completata ---")

if __name__ == "__main__":
    # Downsampling iniziale dei dati HR di test per ottenere i dati LR di test
    load_and_downsample_tif(config.DATA_DIR_TEST_HR, config.DATA_DIR_TEST_LR)
    test()