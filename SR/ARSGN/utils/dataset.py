import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Per il ritaglio funzionale

# --- CONFIGURAZIONE FISSA (da mantenere in config.py) ---
# SCALE_FACTOR = 4 # Fattore di ingrandimento (es. 4x)
# HR_PATCH_SIZE = 512 # Dimensione del ritaglio per l'immagine HR (deve essere multiplo di SCALE_FACTOR)
# --------------------------------------------------------

class SRDataset(Dataset):
    """
    Dataset personalizzato per la Super-Resolution con Ritaglio Casuale di Patch.
    """
    def __init__(self, lr_dir, hr_dir, scale_factor=4, hr_patch_size=512):
        """
        Inizializza il dataset.

        Args:
            lr_dir (str): Percorso della directory contenente le immagini LR.
            hr_dir (str): Percorso della directory contenente le immagini HR (Ground Truth).
            scale_factor (int): Il fattore di ingrandimento S.
            hr_patch_size (int): La dimensione del ritaglio quadrato per le immagini HR.
        """
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        
        if not self.lr_paths or not self.hr_paths:
            raise FileNotFoundError("Assicurati che le directory LR e HR contengano file PNG.")
        
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
        """
        Carica, ritaglia e preprocessa un singolo campione (coppia LR, HR).
        """
        # 1. Carica le immagini
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        lr_img = Image.open(self.lr_paths[idx]).convert("RGB")
        
        # 2. Ottieni i parametri di ritaglio casuale per l'immagine HR
        # Le dimensioni delle immagini (es. 941x1372) vengono ritagliate qui.
        i, j, h, w = transforms.RandomCrop.get_params(hr_img, output_size=(self.hr_patch_size, self.hr_patch_size))
        
        # 3. Applica il ritaglio alla patch HR
        hr_img_crop = TF.crop(hr_img, i, j, h, w)
        
        # 4. Applica il ritaglio alla patch LR
        # Le coordinate devono essere scalate: i_lr = i / S, j_lr = j / S
        i_lr, j_lr = i // self.scale_factor, j // self.scale_factor
        
        # Applica il ritaglio (se le patch LR sono già state pre-generate)
        lr_img_crop = TF.crop(lr_img, i_lr, j_lr, self.lr_patch_size, self.lr_patch_size)

        # 5. Trasforma in tensori
        lr_tensor = self.to_tensor(lr_img_crop)
        hr_tensor = self.to_tensor(hr_img_crop)

        # Verifica di sicurezza (le dimensioni H/W devono ora corrispondere)
        # Se lo script train.py fallisce ancora, è un problema qui o nel modello!
        # if hr_tensor.shape[1] != self.hr_patch_size or lr_tensor.shape[1] != self.lr_patch_size:
        #    raise RuntimeError("Dimensioni della patch non corrette dopo il ritaglio.")

        return lr_tensor, hr_tensor

def get_dataloaders(lr_dir, hr_dir, batch_size, shuffle=True, num_workers=4, 
                    scale_factor=4, hr_patch_size=512):
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