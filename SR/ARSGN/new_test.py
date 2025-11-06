import os
import torch
import torch.nn as nn
import numpy as np
from astropy.io import fits
import sys
import torch.nn.functional as F
from utils.preprocessing import load_and_downsample_tif

# Importa i moduli del tuo progetto
import config
from models.agsr import Net
# Assicurati che load_fits_data sia disponibile nel tuo ambiente
# Ho ri-implementato una versione semplificata per l'uso qui.

# --- FUNZIONI DI UTILITÀ PER FITS ---

def load_fits_data(file_path):
    """Carica i dati FITS e restituisce un tensore (1, H, W) pronto per PyTorch."""
    if not os.path.exists(file_path):
        print(f"ERRORE: File non trovato al percorso: {file_path}")
        return None
    try:
        data = fits.getdata(file_path)
        data = np.squeeze(data).astype(np.float32)
        # Aggiungi una dimensione per il canale (C) e una per il batch (N) -> (1, 1, H, W)
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        return tensor
    except Exception as e:
        print(f"ERRORE durante la lettura del file FITS {file_path}: {e}")
        return None

def save_fits_image(tensor, original_fits_path, output_filename):
    """
    Salva un tensore PyTorch (risultato SR) come nuovo file FITS.
    (La tua implementazione originale è mantenuta, ma adattata per i tensori 4D)
    """
    # Sposta il tensore su CPU e convertilo in array NumPy float32
    # Rimuovi le dimensioni B=1 e C=1, ottenendo (H, W)
    numpy_data = tensor.cpu().squeeze().numpy()
    
    if numpy_data.ndim != 2:
        # Se i dati sono ancora 3D/4D, c'è un problema di shape
        raise ValueError(f"Dati non 2D per il salvataggio FITS. Forma rilevata: {numpy_data.shape}")

    try:
        with fits.open(original_fits_path) as hdul:
            original_header = hdul[0].header
    except Exception as e:
        original_header = fits.Header()

    hdu = fits.PrimaryHDU(numpy_data, header=original_header)
    hdu.header['SR_MODEL'] = ('ARSGN', 'Modello usato per la Super-Risoluzione')
    hdu.header['SR_SCALE'] = (config.SCALE_FACTOR, 'Fattore di ingrandimento applicato')
    hdu.header['COMMENT'] = 'Generated via Super-Resolution (ARSGN) from LR FITS data.'
    hdu.writeto(output_filename, overwrite=True)


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


# --- FUNZIONE PRINCIPALE DI TEST ---

def test():
    """Esegue l'inferenza sull'immagine LR completa utilizzando il Tiling."""
    print(f"--- Avvio Inferenza con Tiling su {config.DEVICE} per dati FITS ---")
    
    # 1. INIZIALIZZAZIONE
    # ----------------------------------------------------------------------
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Simula l'oggetto args (necessario per inizializzare il modello Net(args))
    class Args: pass 
    args = Args()
    
    # Variabili di tiling
    # Assumi che la dimensione del patch sia un parametro nel tuo config
    # Se non specificato, usa un valore comune (es. 64)
    PATCH_SIZE = config.HR_PATCH_SIZE # Dimensione del patch LR usata nell'addestramento
    
    # 2. CARICAMENTO MODELLO
    # ----------------------------------------------------------------------
    # Assicurati che il modello Net possa essere inizializzato correttamente con 'args'
    model = Net(args).to(device)
    
    WEIGHTS_PATH = os.path.join(config.WEIGHTS_DIR, 'model_epoch_100.pth') 
    
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"ERRORE: Pesi del modello non trovati in {WEIGHTS_PATH}")
        
    print(f"Caricamento pesi da: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    # 3. CARICAMENTO ELENCO FILE DI TEST
    # ----------------------------------------------------------------------
    # Utilizza os.listdir per ottenere l'elenco dei file FITS nella directory LR
    lr_files = [f for f in os.listdir(config.DATA_DIR_TEST_LR) if f.endswith('.fits')]
    total_images = len(lr_files)
    
    # 4. CICLO DI INFERENZA IMMAGINE INTERA
    # ----------------------------------------------------------------------
    with torch.no_grad():
        for idx, filename in enumerate(lr_files):
            original_lr_path = os.path.join(config.DATA_DIR_TEST_LR, filename)
            
            print(f"\n--- Elaborazione {filename} ({idx + 1}/{total_images}) ---")
            
            # 4a. Carica l'immagine LR intera (risultato: 1, 1, H, W tensor)
            lr_full_tensor = load_fits_data(original_lr_path)
            
            if lr_full_tensor is None:
                continue # Passa al file successivo in caso di errore di caricamento
            
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
            output_filename = os.path.join(config.OUTPUT_DIR, f"SR_{base_name}_x{config.SCALE_FACTOR}.fits")
            output_ar_filename = os.path.join(config.OUTPUT_DIR, f"AR_{base_name}_x{config.SCALE_FACTOR}.fits")
            
            save_fits_image(
                sr_final_tensor, 
                original_lr_path, # Usa il percorso LR per l'header
                output_filename
            )

            save_fits_image(
                ar_tensor, 
                original_lr_path, 
                output_ar_filename
            )
            
            print(f"-> Salvato: {output_filename}", f"-> Salvato: {output_ar_filename}")

    print("--- Inferenza Completata ---")

if __name__ == "__main__":
    load_and_downsample_tif(config.DATA_DIR_TEST_HR, config.DATA_DIR_TEST_LR)
    test()