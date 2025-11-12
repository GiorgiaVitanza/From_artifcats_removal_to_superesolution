import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from utils.preprocessing import load_and_downsample_tif
from utils.dataset import get_dataloaders   

# Importa i moduli del tuo progetto (Assumi che 'config', 'models.agsr', 'utils.dataset' esistano)
import config 
from models.agsr import Net 


# --- DEFINIZIONE DELLE CLASSI E FUNZIONI DI PERDITA ---

class G_Function(nn.Module):
    """
    Implementa la funzione G(x) = max(x, 0) (ReLU non lineare) per il residuo.
    """
    def forward(self, x):
        return F.relu(x) # Equivalente a x if x > 0, 0 if x < 0

class FFLLoss(nn.Module):
    """
    Implementa la FFL Loss: L1 tra l'ampiezza dello spettro di frequenza
    dell'immagine ricostruita e della Ground Truth.
    """
    def __init__(self):
        super(FFLLoss, self).__init__()
        # Usiamo nn.L1Loss per calcolare la differenza in ampiezza
        self.l1_loss = nn.L1Loss() 

    def forward(self, sr_image, hr_image):
        # 1. Calcola la 2D Fast Fourier Transform (FFT)
        # torch.fft.rfft2 è efficiente per input reali (immagini)
        fft_sr = torch.fft.rfft2(sr_image, dim=(-2, -1), norm="ortho")
        fft_hr = torch.fft.rfft2(hr_image, dim=(-2, -1), norm="ortho")
        
        # 2. Calcola l'ampiezza (Magnitude) dello spettro
        # L'ampiezza è la radice quadrata di (parte reale^2 + parte immaginaria^2)
        # torch.abs calcola la magnitudine di un tensore complesso
        amp_sr = torch.abs(fft_sr)
        amp_hr = torch.abs(fft_hr)
        
        # 3. Calcola la L1 Loss sulla differenza di ampiezza
        # || |\Psi(I^n_{sr})| - |\Psi(I^n_{HR})| ||_1
        loss = self.l1_loss(amp_sr, amp_hr)
        return loss


class ArtifactRemovalSRLoss(nn.Module):
    """
    Funzione di perdita totale L_total = L_art + L_sr + alpha_ffl * L_ffl.
    
    Si assume che:
    - 'artifact_out' sia l'output dell'Artifact Removal Network (I^n_{art}).
    - 'sr_out' sia l'output dell'High-Frequency Generation Network (I^n_{sr}).
    - Il residuo dell'artefatto sia (I^n_{LR} - I^n_{HR}).
    """
    def __init__(self, alpha_ffl=0.1): # alpha_ffl è il peso hyperparameter 
        super(ArtifactRemovalSRLoss, self).__init__()
        
        # Perdite principali
        self.l1_loss = nn.L1Loss() # MAE per L_sr
        self.ffl_loss_fn = FFLLoss() # FFL Loss per L_ffl
        self.g_fn = G_Function() # Funzione G(x) = max(x, 0)
        
        # Hyperparameters
        self.alpha_ffl = alpha_ffl # Peso per L_ffl (dovrebbe essere un valore in config)

    def forward(self, artifact_out, sr_out, lr_image, hr_image):
        """
        Args:
            artifact_out (Tensor): I^n_{art} (output Artifact Removal Network)
            sr_out (Tensor): I^n_{sr} (output High-Frequency Generation Network)
            lr_image (Tensor): I^n_{LR} (Immagine Low Resolution di input)
            hr_image (Tensor): I^n_{HR} (Ground Truth High Resolution)

        Returns:
            Tensor: La perdita totale scalare L_total.
            float: I valori delle perdite componenti per il logging.
        """
        # 0. ALLINEAMENTO DIMENSIONI PER L_art
        # Prende le dimensioni H, W dal target HR (hr_image)
        _, _, H_hr, W_hr = hr_image.shape
        
        # Applica l'upscaling BICUBICO all'immagine LR (lr_image)
        # in modo che abbia la stessa dimensione di hr_image (H_hr, W_hr)
        lr_upscaled = F.interpolate(
            lr_image, 
            size=(H_hr, W_hr), 
            mode='bicubic', 
            align_corners=False # Convenzione comune per l'interpolazione
        )

        # 1. Calcola L_art (Artifact Loss)
        # La sottrazione ora avviene tra due tensori della stessa dimensione (HR x HR)
        residual_target = lr_upscaled - hr_image # I^n_{LR_upscaled} - I^n_{HR}
        g_residual = self.g_fn(residual_target) 
        
        # NOTA: artifact_out DEVE essere già della dimensione HR
        L_art = self.l1_loss(artifact_out, g_residual)
        
        # 2. Calcola L_sr (SR Loss)
        # L_sr = || I^n_{sr} - I^n_{HR} ||_1 (Eq 29)
        L_sr = self.l1_loss(sr_out, hr_image)
        
        # 3. Calcola L_ffl (FFL Loss)
        # L_ffl = || |\Psi(I^n_{sr})| - |\Psi(I^n_{HR})| ||_1 (Eq 30)
        L_ffl = self.ffl_loss_fn(sr_out, hr_image)
        
        # 4. Calcola la Perdita Totale Ponderata
        # L_total = L_art + L_sr + alpha_ffl * L_ffl (Eq 31)
        L_total = L_art + L_sr + (self.alpha_ffl * L_ffl)
        
        return L_total, L_art.item(), L_sr.item(), L_ffl.item()


# --- CICLO DI ADDESTRAMENTO ADATTATO ---

def train():
    
    print(f"--- Avvio Training su {config.DEVICE} ---")
    
    # 1. INIZIALIZZAZIONE
    # ----------------------------------------------------------------------
    device = torch.device(config.DEVICE)
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    
    class Args: pass 
    args = Args()
    
    # 2. MODELLO, DATI, LOSS E OTTIMIZZATORE
    # ----------------------------------------------------------------------
    model = Net(args).to(device)
    
    # Caricamento Dati
    train_loader = get_dataloaders(
        lr_dir=config.DATA_DIR_LR, 
        hr_dir=config.DATA_DIR_HR, 
        batch_size=config.BATCH_SIZE
    )
    
    
    # Funzione di Perdita ADATTATA
    # Nota: Si passa l'hyperparameter alpha_ffl
    criterion = ArtifactRemovalSRLoss(alpha_ffl=config.ALPHA_FFL)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPSILON
    )
    
    scheduler = StepLR(
        optimizer, 
        step_size=config.LR_DECAY_STEP,
        gamma=config.LR_DECAY_FACTOR
    )

    # 3. CICLO DI ADDESTRAMENTO
    # ----------------------------------------------------------------------
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_total_loss = 0
        epoch_L_art = 0
        epoch_L_sr = 0
        epoch_L_ffl = 0
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            # A. Sposta i dati sul dispositivo
            lr_batch = lr_batch.to(device) 
            hr_batch = hr_batch.to(device) 

            # B. Zero i gradienti
            optimizer.zero_grad() 

            # C. Forward Pass: L'output del modello DEVE essere 
            # (artifact_out/I^n_{art}, sr_out/I^n_{sr})
            # Assumiamo che il modello Net(args) restituisca (art_out, sr_out)
            art_out, sr_out = model(lr_batch) 
            
            # D. Calcola la Perdita Totale (Adattata)
            total_loss, L_art_val, L_sr_val, L_ffl_val = criterion(
                art_out, sr_out, lr_batch, hr_batch
            )
            
            # E. Backward Pass
            total_loss.backward()

            # F. Aggiorna i pesi del modello
            optimizer.step() 
            
            # Statistiche
            epoch_total_loss += total_loss.item()
            epoch_L_art += L_art_val
            epoch_L_sr += L_sr_val
            epoch_L_ffl += L_ffl_val
        
        # 4. AGGIORNAMENTO SCHEDULER E LOG
        # ----------------------------------------------------------------------
        scheduler.step()
        
        num_batches = len(train_loader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_L_art = epoch_L_art / num_batches
        avg_L_sr = epoch_L_sr / num_batches
        avg_L_ffl = epoch_L_ffl / num_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Stampa i risultati dell'epoca con le 3 componenti di perdita
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - LR: {current_lr:.6e} - "
              f"Average Loss TOT: {avg_total_loss:.4f} on {num_batches} batches "
              f"(Average L_art: {avg_L_art:.4f}, Average L_sr: {avg_L_sr:.4f}, Average L_ffl: {avg_L_ffl:.4f})")
              
        # 5. SALVATAGGIO MODELLO (Checkpoint)
        # ----------------------------------------------------------------------
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            save_path = os.path.join(config.WEIGHTS_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Modello salvato in {save_path}")

    print("--- Training Completato ---")


if __name__ == '__main__':
    # Esegui train() se non stai usando un notebook.
    # Se i file esterni (config, Net, get_dataloaders_fits) non sono definiti, 
    # usa la configurazione simulata.
    try:
        # load_and_downsample_tif(input_dir=config.DATA_DIR_HR, output_dir=config.DATA_DIR_LR)
        train()
    except NameError as e:
        print(f"Errore: {e}. Assicurati che 'config', 'Net', e 'get_dataloaders_fits' siano definiti.")