import os
import glob
import torch
import numpy as np # Per la manipolazione degli array
from torch.utils.data import Dataset, DataLoader
# from PIL import Image # Non più necessario
# import torchvision.transforms as transforms # Non più necessario per la conversione
# import torchvision.transforms.functional as TF # Non più necessario per il ritaglio funzionale

# --- NUOVE DIPENDENZE PER I FITS ---
try:
    from astropy.io import fits
except ImportError:
    raise ImportError("La libreria 'astropy' è richiesta per leggere i file FITS. Installa con: pip install astropy")
# ------------------------------------

# --- CONFIGURAZIONE FISSA (da mantenere in config.py) ---
# SCALE_FACTOR = 4 # Fattore di ingrandimento (es. 4x)
# HR_PATCH_SIZE = 256 # Dimensione del ritaglio per l'immagine HR (deve essere multiplo di SCALE_FACTOR)
# --------------------------------------------------------

class SRDatasetFITS(Dataset):
    """
    Dataset personalizzato per la Super-Resolution con Ritaglio Casuale di Patch
    per dati astrofisici FITS.
    """
    def __init__(self, lr_dir, hr_dir, scale_factor=4, hr_patch_size=256):
        """
        Inizializza il dataset.

        Args:
            lr_dir (str): Percorso della directory contenente le immagini LR (FITS).
            hr_dir (str): Percorso della directory contenente le immagini HR (Ground Truth FITS).
            scale_factor (int): Il fattore di ingrandimento S.
            hr_patch_size (int): La dimensione del ritaglio quadrato per le immagini HR.
        """
        # Cerca file .fits, non .tif
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.fits')))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.fits')))
        
        if not self.lr_paths or not self.hr_paths:
            raise FileNotFoundError("Assicurati che le directory LR e HR contengano file .fits.")
        
        if len(self.lr_paths) != len(self.hr_paths):
            print("ATTENZIONE: Il numero di immagini LR e HR non corrisponde!")

        # Verifica che la dimensione del ritaglio sia valida
        if hr_patch_size % scale_factor != 0:
            raise ValueError("hr_patch_size deve essere un multiplo esatto di scale_factor.")
        
        self.scale_factor = scale_factor
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale_factor
        
    def __len__(self):
        """Restituisce il numero totale di campioni nel dataset."""
        return len(self.lr_paths)

    def __getitem__(self, idx):
        """
        Carica, ritaglia e preprocessa un singolo campione (coppia LR, HR).
        """
        # 1. Carica le immagini FITS
        with fits.open(self.hr_paths[idx]) as hdul:
            hr_data = hdul[0].data # Potrebbe essere (1, 1, H, W) o (H, W)
        
        with fits.open(self.lr_paths[idx]) as hdul:
            lr_data = hdul[0].data # Potrebbe essere (1, 1, H, W) o (H, W)

        # Aggiunto: Rimuove tutte le dimensioni singole (es. da (1, 1, H, W) a (H, W))
        if hr_data is not None:
             hr_data = np.nan_to_num(hr_data).astype(np.float32).squeeze()
        if lr_data is not None:
             lr_data = np.nan_to_num(lr_data).astype(np.float32).squeeze()
             
        # CONTROLLO: Assicurati che i dati siano 2D dopo lo squeeze
        if hr_data.ndim != 2 or lr_data.ndim != 2:
             raise ValueError(f"Dati FITS non 2D dopo squeeze! Forme: HR={hr_data.shape}, LR={lr_data.shape}")

        
        # 2. Gestione dei dati NaN e conversione a float32 (standard per PyTorch)
        # Sostituiamo i valori NaN con 0 o un altro valore appropriato (dipende dal contesto)
        hr_data = np.nan_to_num(hr_data).astype(np.float32)
        lr_data = np.nan_to_num(lr_data).astype(np.float32)

        # 3. Normalizzazione dei dati
        # L'astrofisica richiede spesso una normalizzazione più complessa (es. Z-score)
        # Qui applichiamo una normalizzazione a [0, 1] semplice, ma potrebbe essere necessario
        # modificarla in base al range di flusso del tuo dataset.
        data_max = max(hr_data.max(), lr_data.max())
        if data_max > 0:
            hr_data /= data_max
            lr_data /= data_max
        
        # 4. Ottieni i parametri di ritaglio casuale per l'immagine HR
        H, W = hr_data.shape
        
        # Calcolo casuale del punto di inizio (top, left)
        # Assicurati che H e W siano maggiori o uguali a hr_patch_size
        if H < self.hr_patch_size or W < self.hr_patch_size:
             raise ValueError(f"L'immagine HR {self.hr_paths[idx]} ({H}x{W}) è troppo piccola per il ritaglio di dimensione {self.hr_patch_size}.")
             
        i = np.random.randint(0, H - self.hr_patch_size + 1) # riga di partenza (top)
        j = np.random.randint(0, W - self.hr_patch_size + 1) # colonna di partenza (left)

        # 5. Applica il ritaglio alla patch HR (numpy slicing)
        hr_patch = hr_data[i : i + self.hr_patch_size, j : j + self.hr_patch_size]
        
        # 6. Applica il ritaglio alla patch LR (numpy slicing)
        # Le coordinate devono essere scalate: i_lr = i / S, j_lr = j / S
        i_lr, j_lr = i // self.scale_factor, j // self.scale_factor
        
        # Le patch LR devono essere già allineate e di dimensione (H/S, W/S) rispetto alle HR
        lr_patch = lr_data[i_lr : i_lr + self.lr_patch_size, j_lr : j_lr + self.lr_patch_size]
        
        # 7. Trasforma in tensori PyTorch e aggiungi la dimensione del canale (C=1)
        # Da (H, W) a (C, H, W) -> (1, H, W)
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)

        # 8. Verifica di sicurezza (facoltativa)
        if hr_tensor.shape[1] != self.hr_patch_size or lr_tensor.shape[1] != self.lr_patch_size:
           raise RuntimeError(f"Dimensioni della patch non corrette: HR={hr_tensor.shape}, LR={lr_tensor.shape}. Previsto: HR=(1,{self.hr_patch_size},{self.hr_patch_size}), LR=(1,{self.lr_patch_size},{self.lr_patch_size}).")

        return lr_tensor, hr_tensor

# ------------------------------------------------------
## Funzione di Utility per il DataLoader
# ------------------------------------------------------

def get_dataloaders_fits(lr_dir, hr_dir, batch_size, shuffle=True, num_workers=0, 
                         scale_factor=4, hr_patch_size=256):
    """
    Crea e restituisce il DataLoader per l'addestramento con dati FITS.
    """
    
    # 1. Crea l'istanza del Dataset (versione FITS)
    sr_dataset = SRDatasetFITS(lr_dir, hr_dir, scale_factor, hr_patch_size)

    # 2. Crea l'istanza del DataLoader
    data_loader = DataLoader(
        sr_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=False # Mantienilo False o True a seconda della tua configurazione GPU
    )
    
    return data_loader