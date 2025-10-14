import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class SRDataset(Dataset):
    """
    Dataset per la Super-Risoluzione (SR).
    Carica immagini ad alta risoluzione (HR) e genera dinamicamente
    le immagini a bassa risoluzione (LR) tramite downsampling bicubico.
    """

    def __init__(self, hr_dir, scale_factor=4, lr_downsampling_method='bicubic', patch_size=None, transform=None):
        """
        Inizializza il dataset.

        Args:
            hr_dir (str): Percorso alla directory contenente le immagini HR.
            scale_factor (int): Fattore di scala desiderato (es. 4x).
            lr_downsampling_method (str): Metodo di downsampling. Deve essere 'bicubic'.
            patch_size (int, optional): Dimensione del ritaglio per l'addestramento. Se None, usa l'immagine intera.
            transform (callable, optional): Trasformazioni aggiuntive da applicare all'immagine HR.
        """
        
        # --- Parametri obbligatori e validazione ---
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        
        if lr_downsampling_method.lower() != 'bicubic':
            raise ValueError("Il parametro 'lr_downsampling_method' deve essere 'bicubic' come specificato.")
        
        # Mappatura dei metodi di resampling PIL
        self.resample_method = Image.BICUBIC

        # Lista di tutti i percorsi delle immagini (supporta .png, .jpg, .bmp)
        self.image_paths = sorted(
            glob.glob(os.path.join(hr_dir, '*.[pP][nN][gG]')) +
            glob.glob(os.path.join(hr_dir, '*.[jJ][pP][gG]')) +
            glob.glob(os.path.join(hr_dir, '*.[bB][mM][pP]'))
        )
        
        if not self.image_paths:
            raise FileNotFoundError(f"Nessuna immagine trovata nella directory: {hr_dir}")

        # --- Trasformazioni ---
        # ToTensor() converte l'immagine PIL in un tensore PyTorch (C x H x W)
        # e normalizza i valori [0, 255] a [0.0, 1.0].
        self.to_tensor = ToTensor()
        self.transform = transform


    def __len__(self):
        """Restituisce il numero totale di immagini."""
        return len(self.image_paths)


    def _random_crop(self, hr_img, patch_size):
        """Esegue un ritaglio casuale coerente su HR e LR (per il training)."""
        
        w, h = hr_img.size
        
        # Assicurati che l'immagine sia abbastanza grande
        if w < patch_size or h < patch_size:
            # Se l'immagine è più piccola, la ridimensioniamo o la saltiamo (qui la ridimensioniamo)
            # Nota: In un dataset reale, potresti voler filtrare queste immagini.
            hr_img = hr_img.resize((patch_size, patch_size), Image.BICUBIC)
            w, h = patch_size, patch_size
        
        # Calcola le coordinate del ritaglio per l'immagine HR
        x = np.random.randint(0, w - patch_size + 1)
        y = np.random.randint(0, h - patch_size + 1)
        
        # Ritorna il ritaglio
        return hr_img.crop((x, y, x + patch_size, y + patch_size))


    def _downsample_bicubic(self, hr_img, scale_factor):
        """
        Genera l'immagine a bassa risoluzione (LR) utilizzando il downsampling bicubico
        come specificato.
        """
        w, h = hr_img.size
        
        # Calcola le dimensioni di output LR
        lr_w = w // scale_factor
        lr_h = h // scale_factor
        
        # Downsampling usando la funzione bicubica (PIL.Image.BICUBIC)
        lr_img = hr_img.resize((lr_w, lr_h), self.resample_method)
        
        # Il modello di solito richiede l'immagine LR di input, non l'immagine 
        # ingrandita a bicubica per matchare le dimensioni HR.

        return lr_img

    
    def __getitem__(self, index):
        """Restituisce la coppia immagine LR (input) e HR (target)."""
        
        # 1. Carica l'immagine HR
        hr_path = self.image_paths[index]
        hr_img = Image.open(hr_path).convert('RGB') # Assicurati che sia RGB
        
        # 2. Ritaglio casuale (se il patch_size è definito)
        if self.patch_size is not None:
            hr_img = self._random_crop(hr_img, self.patch_size)
            
        # 3. Applicazione di trasformazioni aggiuntive (se definite)
        if self.transform is not None:
            hr_img = self.transform(hr_img)

        # 4. Generazione dell'immagine LR tramite downsampling
        lr_img = self._downsample_bicubic(hr_img, self.scale_factor)
        
        # 5. Conversione in tensori PyTorch
        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)
        
        # Ritorna il tensore LR (input) e il tensore HR (target)
        return lr_tensor, hr_tensor

# --- Esempio di Utilizzo (Opzionale, per test) ---
if __name__ == '__main__':
    # Creare una cartella 'test_images' e metterci qualche immagine.
    # Assicurati di installare le librerie: pip install torch torchvision pillow
    
    # Esempio di utilizzo:
    try:
        # Crea un'istanza del dataset (NOTA: Cambia il percorso)
        # Se esegui lo script dalla cartella ARSGN, '..' potrebbe essere appropriato
        dataset = SRDataset(
            hr_dir='/path/to/your/HR/images',  # <<< CAMBIA QUESTO PERCORSO
            scale_factor=4,
            patch_size=128 # Rimuovi per usare l'immagine intera
        )
        
        print(f"Numero totale di immagini nel dataset: {len(dataset)}")
        
        # Accedi al primo elemento
        lr_tensor, hr_tensor = dataset[0]
        
        print("\nInformazioni sull'output del primo elemento:")
        print(f"Dimensione LR Tensor (Input): {lr_tensor.shape}") # C x H/s x W/s
        print(f"Dimensione HR Tensor (Target): {hr_tensor.shape}") # C x H x W
        
        # Esempio di verifica della dimensione (se patch_size=128 e scale_factor=4)
        # Output atteso: LR: [3, 32, 32], HR: [3, 128, 128]
        
    except FileNotFoundError as e:
        print(f"ERRORE: {e}. Assicurati di avere il percorso corretto e immagini nella directory.")
    except Exception as e:
        print(f"Si è verificato un errore durante il test: {e}")