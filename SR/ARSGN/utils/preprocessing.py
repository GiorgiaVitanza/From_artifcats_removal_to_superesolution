from skimage.io import imread, imsave # Nuove importazioni per TIFF
import numpy as np
import os
import sys

from skimage.transform import resize

# 1. Definisce la cartella radice del progetto: sale da 'utils' a 'ARSGN'
# (o dalla cartella di utilità alla cartella che contiene config.py)
percorso_arsgn = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. Aggiunge il percorso di ARSGN a sys.path
if percorso_arsgn not in sys.path:
    sys.path.append(percorso_arsgn)

# --- Fine del codice di aggiunta percorso ---
import config


def load_and_downsample_tif(input_dir=config.DATA_DIR_HR, output_dir=config.DATA_DIR_LR):
    """
    Carica un file TIFF, lo downsample e salva la versione a bassa risoluzione
    (LR) come un nuovo file TIFF.
    """
    os.makedirs(output_dir, exist_ok=True)

    S = config.SCALE_FACTOR # Fattore di scala (es. 2, 4)

    for filename in os.listdir(input_dir):
        # *** Cambiato da '.fits' a '.tif' o '.tiff' ***
        if filename.endswith(('.tif', '.tiff')):
            hr_path = os.path.join(input_dir, filename)
            # Cambia l'estensione del file di output (se necessario, ma qui la manteniamo '.tif')
            # Se vuoi cambiare l'estensione in .png, usa: filename.replace('.tif', '_lr.png')
            lr_path = os.path.join(output_dir, filename)

            if not os.path.isfile(hr_path):
                continue
            else:
                print(f"{hr_path} è un file TIFF valido. Procedo con l'elaborazione.")

            try:
                # *** Carica il file TIFF con imread ***
                hr_data = imread(hr_path).astype(np.float32)
                print(f"Forma dati HR caricati: {hr_data.shape}")

                # Gestisce array con più dimensioni (es. (H, W, Canali) o (1, H, W))
                # Se è un'immagine a colori o ha una dimensione extra, assicurati di downsamplare solo H e W.
                if hr_data.ndim == 3:
                    # Rileva se il canale è la prima o l'ultima dimensione
                    # Si presume (H, W, C) o (C, H, W). Assumiamo (H, W, C) per semplicità
                    H, W, C = hr_data.shape
                    new_H, new_W = H // S, W // S
                    new_shape = (new_H, new_W, C)
                elif hr_data.ndim == 2:
                    H, W = hr_data.shape
                    new_H, new_W = H // S, W // S
                    new_shape = (new_H, new_W)
                else:
                    print(f"AVVISO: Immagine con dimensione non gestita ({hr_data.ndim}). Ignoro.")
                    continue

                # Pulisci NaN (se presenti, anche se meno comuni in immagini non FITS)
                new_data = np.nan_to_num(hr_data, nan=0.0)
                print(f"Forma dati da downsampling: {new_data.shape}")
                # Esegui il downscaling (interpolazione bilineare/bicubica)
                # L'ordine=3 corrisponde all'interpolazione bicubica
                lr_data = resize(
                    new_data,
                    new_shape, # Usa la nuova forma calcolata
                    order=3,
                    mode='reflect',
                    anti_aliasing=True,
                    preserve_range=True # Importante per mantenere i valori originali dei pixel
                ).astype(np.float32)

                print(f"Forma dati LR generati: {lr_data.shape}")


                # 1. Normalizza i dati (Scalatura tra 0.0 e 1.0)
                # Trova il massimo valore di pixel nel dataset originale (o usa un valore noto, es. 65535)
                # **ATTENZIONE: Se hai già dati normalizzati (0.0-1.0), salta questa riga!**
                max_val = np.max(hr_data) # O np.max(lr_data) - usa il massimo dell'HR per coerenza
                lr_data_normalized = lr_data / max_val
                # La normalizzazione è critica!
                print(f"Valori LR normalizzati (min/max): {np.min(lr_data_normalized):.4f} / {np.max(lr_data_normalized):.4f}")


                # 2. Converte in un tipo di intero standard (es. 16-bit)
                # Questo moltiplica l'array normalizzato per il massimo valore del tipo intero (es. 65535 per uint16)
                # e lo converte in uint16.
                # Scegli uint16 se i dati HR originali erano a 16 bit, altrimenti usa uint8 (max 255)
                # In molti casi, le immagini scientifiche sono 16-bit.
                lr_data_int = (lr_data_normalized * 65535).astype(np.uint16)
                

                # *** Salva il nuovo file TIFF con imsave ***
                # Salva l'array di interi invece del float32 non normalizzato
                imsave(lr_path, lr_data_int)
                print(f"Creato LR: {lr_data_int.shape} (dtype: {lr_data_int.dtype}) per HR: {hr_data.shape}. Salvato in {lr_path}")

            except Exception as e:
                print(f"ERRORE nell'elaborazione del file {filename}: {e}")
                continue

        else:
            print(f"File non TIFF ({filename}). Ignoro.")
            continue

    print("Rigenerazione dei file LR completata.")