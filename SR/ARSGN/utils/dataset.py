import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Per il ritaglio funzionale
import config

# --- CONFIGURAZIONE FISSA (da mantenere in config.py) ---
# SCALE_FACTOR = 4 # Fattore di ingrandimento (es. 4x)
# HR_PATCH_SIZE = 256 # Dimensione del ritaglio per l'immagine HR (deve essere multiplo di SCALE_FACTOR)
# --------------------------------------------------------

class SRDataset(Dataset):
    """
    Dataset personalizzato per la Super-Resolution con Ritaglio Casuale di Patch.
    """
    
    def __init__(self, lr_dir, hr_dir, scale_factor=config.SCALE_FACTOR, hr_patch_size=config.HR_PATCH_SIZE):
        """
        Inizializza il dataset.

        Args:
            lr_dir (str): Percorso della directory contenente le immagini LR.
            hr_dir (str): Percorso della directory contenente le immagini HR (Ground Truth).
            scale_factor (int): Il fattore di ingrandimento S.
            hr_patch_size (int): La dimensione del ritaglio quadrato per le immagini HR.
        """
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.tif')))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.tif')))
        
        if not self.lr_paths or not self.hr_paths:
            raise FileNotFoundError("Assicurati che le directory LR e HR contengano file tif.")
        
        if len(self.lr_paths) != len(self.hr_paths):
            print("ATTENZIONE: Il numero di immagini LR e HR non corrisponde!")

        # Verifica che la dimensione del ritaglio sia valida
        if hr_patch_size % scale_factor != 0:
            raise ValueError("hr_patch_size deve essere un multiplo esatto di scale_factor.")
        
        self.scale_factor = scale_factor
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale_factor
        
        # Trasformazione: converte l'immagine PIL in un tensore e normalizza a [0.0, 1.0]
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        """Restituisce il numero totale di campioni nel dataset."""
        return len(self.lr_paths)


    def __getitem__(self, idx):
        # 1. Carica le immagini HR e LR come NumPy float32
        # Usiamo imread (da skimage) perché gestisce meglio i TIFF float32 rispetto a PIL
        hr_img_np = imread(self.hr_paths[idx]).astype(np.float32)
        lr_img_np = imread(self.lr_paths[idx]).astype(np.float32)
        
        # Subito dopo: hr_img_np = imread(self.hr_paths[idx]).astype(np.float32)
        print(f"DEBUG: Max value of HR image loaded: {np.max(hr_img_np)}")
        print(f"DEBUG: Min value of HR image loaded: {np.min(hr_img_np)}")
        # Dovresti vedere un max di circa 184.0, non 1.0!

        # 2. Converte in tensori PyTorch (H, W, C) -> (C, H, W)
        hr_tensor = torch.from_numpy(hr_img_np).permute(2, 0, 1) 
        lr_tensor = torch.from_numpy(lr_img_np).permute(2, 0, 1)

        # 3. Trova il massimo per la normalizzazione
        # Usiamo max() qui, ma in un training reale è meglio usare un valore fisso (es. 255 o 65535) 
        # se il range è noto, altrimenti il batch sarà non uniformemente normalizzato.
        max_val_hr = hr_tensor.max().item()
        if max_val_hr < 1e-4: max_val_hr = 1.0 

        # 4. **NORMALIZZAZIONE ESPLICITA**
        lr_tensor_norm = lr_tensor / max_val_hr
        hr_tensor_norm = hr_tensor / max_val_hr
        
        # 5. Esegui il Cropping Casuale (come hai fatto prima) sui tensori normalizzati
        # Nota: transforms.RandomCrop.get_params richiede la dimensione H e W
        H, W = hr_tensor_norm.shape[1:] 
        
        # Calcolo casuale (assicurati che sia entro i limiti)
        i = torch.randint(0, H - self.hr_patch_size + 1, (1,)).item()
        j = torch.randint(0, W - self.hr_patch_size + 1, (1,)).item()
        
        # Applicazione del Cropping
        lr_patch = lr_tensor_norm[:, 
                                i // self.scale_factor : i // self.scale_factor + self.lr_patch_size, 
                                j // self.scale_factor : j // self.scale_factor + self.lr_patch_size]
                                
        hr_patch = hr_tensor_norm[:, i : i + self.hr_patch_size, j : j + self.hr_patch_size]
        
        # Verifica finale (DEBUG)
        print(f"Patch HR shape: {hr_patch.shape}, Patch LR shape: {lr_patch.shape}")
        

        return lr_patch, hr_patch # Ritorna le patch normalizzate e ritagliate

def get_dataloaders(lr_dir, hr_dir, batch_size = config.BATCH_SIZE, shuffle=True, num_workers=0, 
                    scale_factor=config.SCALE_FACTOR, hr_patch_size=config.HR_PATCH_SIZE):
    """
    Crea e restituisce il DataLoader per l'addestramento.
    """
    
    # 1. Crea l'istanza del Dataset
    sr_dataset = SRDataset(lr_dir, hr_dir, scale_factor, hr_patch_size)

    # 2. Crea l'istanza del DataLoader
    data_loader = DataLoader(
        sr_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=False # <-- MODIFICATO: disabilitato se non si usa GPU
    )
    
    return data_loader