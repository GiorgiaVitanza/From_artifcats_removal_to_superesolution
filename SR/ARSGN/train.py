import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Importa i moduli del tuo progetto (Assumi che 'models' e 'utils' siano nella stessa directory)
import config
from models.agsr import Net 
from utils.dataset import get_dataloaders 

# --- CONFIGURAZIONE DELLA PERDITA ---
# La perdita totale sarà una combinazione dell'output dello Stage 1 (sr_1) e dello Stage 2 (sr_2)
# È comune assegnare un peso maggiore all'output finale (sr_2).

def calculate_total_loss(sr_1, sr_2, hr_batch, criterion):
    """
    Calcola la perdita combinata per il modello a due stadi.

    Args:
        sr_1 (Tensor): Output Super-Resolution dello Stage 1.
        sr_2 (Tensor): Output Super-Resolution dello Stage 2 (Finale).
        hr_batch (Tensor): Immagine High Resolution (Ground Truth).
        criterion (Loss Function): Funzione di perdita (es. L1Loss).

    Returns:
        Tensor: La perdita totale combinata.
    """
    # Esempio di peso: 40% sul primo stadio, 60% sul secondo (o 50/50)
    WEIGHT_STAGE1 = 0.4
    WEIGHT_STAGE2 = 0.6 
    
    # 1. Perdita sullo Stage 1
    loss_stage1 = criterion(sr_1, hr_batch)
    
    # 2. Perdita sullo Stage 2 (Output Finale)
    loss_stage2 = criterion(sr_2, hr_batch)
    
    # 3. Perdita Totale Ponderata
    total_loss = (WEIGHT_STAGE1 * loss_stage1) + (WEIGHT_STAGE2 * loss_stage2)
    
    return total_loss, loss_stage1.item(), loss_stage2.item()


def train():
    print(f"--- Avvio Training su {config.DEVICE} ---")
    
    # 1. INIZIALIZZAZIONE
    # ----------------------------------------------------------------------
    device = torch.device(config.DEVICE)
    
    # Crea la directory per salvare i pesi se non esiste
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    
    # Simula l'oggetto args richiesto dalla classe Net
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
    
    # Funzione di Perdita (Mean Absolute Error o L1 Loss è comune in SR)
    criterion = nn.L1Loss() 
    
    # Ottimizzatore Adam con i parametri specificati
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPSILON
    )
    
    # Scheduler del Learning Rate (Decadimento)
    # Il testo indica: "diminuisce di un fattore di 10 per ogni 500 epoche"
    scheduler = StepLR(
        optimizer, 
        step_size=config.LR_DECAY_STEP, # 500
        gamma=config.LR_DECAY_FACTOR    # 0.1 (1/10)
    )

    # 3. CICLO DI ADDESTRAMENTO
    # ----------------------------------------------------------------------
    for epoch in range(config.NUM_EPOCHS):
        model.train()  # Imposta il modello in modalità training
        epoch_total_loss = 0
        epoch_loss1 = 0
        epoch_loss2 = 0
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            # A. Sposta i dati sul dispositivo (GPU/CPU)
            lr_batch = lr_batch.to(device) 
            hr_batch = hr_batch.to(device) 

            # B. Zero i gradienti
            optimizer.zero_grad() 

            # C. Forward Pass: l'output sono i due stadi (sr_1, sr_2)
            sr_1, sr_2 = model(lr_batch) 
            
            # D. Calcola la Perdita Totale
            total_loss, loss1, loss2 = calculate_total_loss(sr_1, sr_2, hr_batch, criterion)
            
            # E. Backward Pass
            total_loss.backward()

            # F. Aggiorna i pesi del modello
            optimizer.step() 
            
            # Statistiche
            epoch_total_loss += total_loss.item()
            epoch_loss1 += loss1
            epoch_loss2 += loss2
        
        # 4. AGGIORNAMENTO SCHEDULER E LOG
        # ----------------------------------------------------------------------
        
        # Aggiorna il learning rate (lo scheduler controlla il timing)
        scheduler.step()
        
        # Calcola le perdite medie per l'epoca
        num_batches = len(train_loader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_loss1 = epoch_loss1 / num_batches
        avg_loss2 = epoch_loss2 / num_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Stampa i risultati dell'epoca
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - LR: {current_lr:.6e} - "
              f"Loss Tot: {avg_total_loss:.4f} (Stage 1: {avg_loss1:.4f}, Stage 2: {avg_loss2:.4f})")
              
        # 5. SALVATAGGIO MODELLO (Checkpoint)
        # ----------------------------------------------------------------------
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            save_path = os.path.join(config.WEIGHTS_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Modello salvato in {save_path}")

    print("--- Training Completato ---")


if __name__ == '__main__':
    train()