import torch
import torch.nn as nn
from agsr import Net # Assumi che il tuo codice Net sia in 'model.py'
import os
from datetime import datetime

# --- 1. Definizione di una classe args fittizia (necessaria per Net) ---
class DummyArgs:
    pass

# --- 2. Inizializzazione del Modello ---
args = DummyArgs()
# Istanzia il modello Net (i pesi sono inizializzati casualmente)
model = Net(args) 

# --- 3. Preparazione di Altri Stati (tipici di un checkpoint) ---
# Di solito, un checkpoint salva lo stato dell'ottimizzatore, 
# la loss migliore, l'epoca corrente, ecc.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epoch = 10
best_loss = 0.05
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 4. Creazione del Dizionario di Checkpoint ---
# PyTorch di solito salva un dizionario che contiene il 'state_dict' del modello.
checkpoint = {
    'epoch': epoch,
    'best_loss': best_loss,
    'model_state_dict': model.state_dict(), # <--- QUESTO È IL CUORE DEL CHECKPOINT (I PESI)
    'optimizer_state_dict': optimizer.state_dict(),
    'timestamp': timestamp,
    'description': 'Checkpoint salvato dopo 10 epoche di addestramento.'
}

# --- 5. Salvataggio del File ---
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'simulated_weights.pth')

try:
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"File di checkpoint simulato salvato con successo in: {CHECKPOINT_PATH}")

    # Esempio di come leggere e ispezionare il contenuto del file
    loaded_checkpoint = torch.load(CHECKPOINT_PATH)
    print("\n--- Contenuto del Checkpoint Caricato ---")
    print(f"Epoca: {loaded_checkpoint['epoch']}")
    print(f"Migliore Loss: {loaded_checkpoint['best_loss']}")
    
    # Stampa un esempio di chiave e dimensione del tensore dei pesi
    print(f"Numero di tensori del modello salvati: {len(loaded_checkpoint['model_state_dict'])}")
    for name, param in loaded_checkpoint['model_state_dict'].items():
        if 'head_1.weight' in name:
            print(f"  Esempio di tensore (head_1.weight): Dimensione {param.size()}")
            break

except Exception as e:
    print(f"Errore durante la creazione/salvataggio del checkpoint: {e}")