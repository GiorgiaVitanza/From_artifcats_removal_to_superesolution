import numpy as np
from skimage.metrics import structural_similarity as ssim
# Importiamo imread da skimage.io per caricare i file TIFF
from skimage.io import imread 
import os
import config
import matplotlib.pyplot as plt

# Rimosso: from astropy.io import fits (Non più necessaria)


# --- Funzioni Ausiliarie (Invariate o Leggermente Modificate) ---

def align_and_crop_to_min_shape(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ritaglia due array NumPy (immagini) per farli corrispondere alla dimensione 
    minima tra i due, a partire dall'angolo in alto a sinistra (0, 0).
    (Funzione lasciata invariata in quanto agisce su array NumPy generici).
    """
    
    # 1. Ottieni le forme (dimensioni) delle due immagini
    shape1 = img1.shape
    shape2 = img2.shape
    
    # Verifica che gli array abbiano almeno 2 dimensioni (altezza, larghezza)
    if len(shape1) < 2 or len(shape2) < 2:
        raise ValueError("Le immagini devono avere almeno due dimensioni (altezza e larghezza).")

    # 2. Determina le dimensioni minime comuni
    min_h = min(shape1[0], shape2[0])
    min_w = min(shape1[1], shape2[1])
    
    # Se le immagini sono a colori (3 dimensioni: H, W, C), assicurati che anche i canali corrispondano
    if len(shape1) > 2 and len(shape2) > 2:
        min_c = min(shape1[2], shape2[2])
        
        # 3. Applica il ritaglio a entrambe le immagini
        cropped_img1 = img1[:min_h, :min_w, :min_c]
        cropped_img2 = img2[:min_h, :min_w, :min_c]
        
    else:
        # Immagini in scala di grigi (2 dimensioni: H, W)
        cropped_img1 = img1[:min_h, :min_w]
        cropped_img2 = img2[:min_h, :min_w]
        
    return cropped_img1, cropped_img2


def calculate_psnr(img1, img2):
    """
    Calcola il Peak Signal-to-Noise Ratio (PSNR) tra due immagini.
    (Funzione lasciata invariata, PSNR si basa su MSE che è indipendente dal formato).
    """
    # Converti le immagini in float per i calcoli
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Calcola il Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        print("Le immagini sono identiche; PSNR è infinito.")
        return float('inf')
    
    # max_val viene calcolato dinamicamente, il che è appropriato per dati scientifici float
    max_val = max(np.max(img1), np.max(img2))
    
    if max_val == 0:
        print("Attenzione: l'immagine originale è completamente nera. PSNR non calcolabile.")
        return 0.0 # Ritorna 0.0 o un altro indicatore di non calcolabile
    else:
        print(f"Valore massimo dei pixel: {max_val:.4f}")
        
    # Calcola il PSNR in base alla formula
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr


def load_tif_data(file_path):
    """
    Carica i dati dell'immagine da un file TIFF utilizzando skimage.io.
    Converte i dati in np.float32 per i calcoli.
    
    :param file_path: Il percorso completo del file TIFF.
    :return: Un array NumPy contenente i dati dell'immagine, o None in caso di errore.
    """
    if not os.path.exists(file_path):
        print(f"ERRORE: File non trovato al percorso: {file_path}")
        return None
        
    try:
        # imread carica i dati dell'immagine TIFF
        data = imread(file_path)
        # Assicuriamo che sia float32 per coerenza con il modello e i calcoli.
        return data.astype(np.float32)
    except Exception as e:
        print(f"ERRORE durante la lettura del file TIFF {file_path}: {e}")
        return None

# -----------------------------------------------------

def main_fun():
    """
    Funzione principale per caricare immagini TIFF, calcolare PSNR e SSIM.
    """
    
    data_dir_hr = config.DATA_DIR_TEST_HR
    output_dir = config.OUTPUT_DIR
    file_extension = '.tif'
    

    # 1. Caricamento delle Immagini TIFF
    # Cerca file con estensione .tif o .tiff (gestendo il case)
    hr_files = [f for f in os.listdir(data_dir_hr) if f.lower().endswith(('.tif', '.tiff'))]
    
    if not hr_files:
        print(f"ATTENZIONE: Nessun file TIFF trovato nella directory {data_dir_hr}")
        return

    total_psnr = 0.0
    total_ssim = 0.0
    valid_count = 0

    for idx, hr_filename in enumerate(hr_files):
        
        # 1a. Carica Immagine HR (Ground Truth)
        hr_image = load_tif_data(os.path.join(data_dir_hr, hr_filename))
        
        # 1b. Determina il nome del file SR (Super Resolution) e caricalo
        # Esempio: "ImmagineA.tif" -> cerca "SR_ImmagineA_x4.tif"
        base_name, _ = os.path.splitext(hr_filename)
        sr_filename = f"SR_{base_name}_x{config.SCALE_FACTOR}{file_extension}"
        sr_image = load_tif_data(os.path.join(output_dir, sr_filename))
        
        if hr_image is None or sr_image is None:
            print(f"Skipping {hr_filename} a causa di un errore di caricamento.")
            continue
        

        # 2. Gestione NaN e Allineamento
        # ---
        # Se i dati TIFF non contengono NaN (più comune), queste righe sono meno critiche,
        # ma le manteniamo per robustezza.
        
        # Calcola la media solo sui valori NON-NaN
        mean_val_hr = np.nanmean(hr_image)
        mean_val_sr = np.nanmean(sr_image)

        # Sostituisci tutti i valori NaN con la media calcolata
        img_1_cleaned = np.where(np.isnan(hr_image), mean_val_hr, hr_image)
        img_2_cleaned = np.where(np.isnan(sr_image), mean_val_sr, sr_image)
        
        # Allinea le immagini alla dimensione comune minima prima del calcolo
        img_1, img_2 = align_and_crop_to_min_shape(img_1_cleaned, img_2_cleaned)
        
        # 3. Calcolo e Visualizzazione dei Risultati
        # ---
        print(f"\nValutazione delle metriche per il file: {hr_filename}"+f" ({idx+1}/{len(hr_files)})")
        
        try:
            # Calcolo PSNR
            psnr = calculate_psnr(img_1, img_2)
            print(f"PSNR: {psnr:.2f} dB")
            total_psnr += psnr
        except Exception as e:
            print(f"Errore nel calcolo del PSNR per {hr_filename}: {e}")
            psnr = 0.0

        try:
            # Calcolo SSIM
            # Il parametro channel_axis è necessario se le immagini sono a colori (3D)
            channel_axis = -1 if img_1.ndim == 3 else None
            
            # Nota sul data_range: img_1.max() - img_1.min() è corretto per i dati float
            ssim_value, _ = ssim(img_1, img_2, 
                                 data_range=img_1.max() - img_1.min(), 
                                 channel_axis=channel_axis, # Aggiunto per gestire RGB
                                 full=True)
            print(f"SSIM: {ssim_value:.4f}")
            total_ssim += ssim_value
            valid_count += 1
        except Exception as e:
            print(f"Errore nel calcolo dell'SSIM per {hr_filename}: {e}")
            ssim_value = 0.0

       

    # 4. Calcolo e Stampa delle Medie
    # ---
    if valid_count > 0:
        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        print("\n" + "="*40)
        print(f"Riepilogo delle metriche su {valid_count} file:")
        print(f"Media PSNR: {avg_psnr:.2f} dB")
        print(f"Media SSIM: {avg_ssim:.4f}")
        print("="*40)
    else:
        print("\nNessuna metrica calcolata con successo.")
        

if __name__ == '__main__':
    print("\nValutazione del dataset tif:")
    main_fun()