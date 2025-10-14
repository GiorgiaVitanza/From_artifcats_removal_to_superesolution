import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Le seguenti importazioni sono placeholder.
# Assicurati che i file 'model.py' (con la classe ARSGN) e 'dataset.py'
# (con la classe SRDataset) siano presenti nella cartella ARSGN.
from agsr import *# <<<<<< AGGIORNATO: ARSGN
from common import *
from dataset import SRDataset

# ... (all'inizio di train.py) ...
# Importa il modulo 'collections'
import collections 

# --- Aggiungi una classe/oggetto per simulare gli argomenti di configurazione ---
# (Sostituisci i valori con quelli effettivi richiesti dal tuo modello Net)
class Config:
    def __init__(self):
        # Questi sono solo ESEMPI. Devi trovare i valori corretti in model.py o nel codice originale.
        self.scale = 4           # Esempio: Fattore di Super-Risoluzione (2, 3, 4, ecc.)
        self.n_colors = 3        # Esempio: Numero di canali (3 per RGB)
        self.n_feats = 64        # Esempio: Numero di feature maps intermedie
        self.n_resblocks = 16    # Esempio: Numero di blocchi residui
        # Aggiungi qui qualsiasi altro parametro richiesto da Net.__init__(self, args)

# ... (all'interno della funzione main() in train.py) ...

# --- 2. Inizializzazione Modello e Loss ---
# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 1. Crea l'oggetto argomenti
    args = Config() 
    
    # 2. Passa l'oggetto al modello
    # Sostituito il nome del modello da ARSGN a Net come nel traceback
    model = Net(args).to(device) 
    
except NameError:
    print("ERRORE: La classe Net non è definita. Assicurati che 'model.py' esista nella cartella ARSGN.")
    
# --- Parametri di Training specificati ---
BATCH_SIZE_PER_GPU = 16
N_GPUS = 3  # Tre Nvidia TITAN GPU
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * N_GPUS # 16 * 3 = 48 (se si usa DataParallel)

# Parametri dell'Ottimizzatore Adam
INITIAL_LR = 1e-4
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8 # Corrisponde a eta = 1e-8

# Schedulazione del Learning Rate
LR_DECAY_EPOCHS = 500
LR_DECAY_FACTOR = 10
MAX_EPOCHS = 2000 # Assunzione di un numero massimo di epoche

# --- Setup Ambiente ---
print("Ambiente richiesto: Ubuntu 18.04, CUDA 10.2, CUDNN 7.5, 3x Nvidia TITAN GPUs.")

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and torch.cuda.device_count() >= N_GPUS:
    print(f"Trovate {torch.cuda.device_count()} GPU. Utilizzo {N_GPUS} per DataParallel.")
elif torch.cuda.is_available():
    print(f"ATTENZIONE: Trovate solo {torch.cuda.device_count()} GPU. Il setup originale ne prevedeva {N_GPUS}.")
else:
    print("ATTENZIONE: GPU non disponibile. Training su CPU.")


# --- Funzione per la Schedulazione del Learning Rate ---
def adjust_learning_rate(optimizer, epoch):
    """
    Diminuisce il learning rate di un fattore 10 ogni 500 epoche.
    """
    lr = INITIAL_LR * (1.0 / LR_DECAY_FACTOR ** (epoch // LR_DECAY_EPOCHS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    # --- 1. Preparazione Dataset ---
    # Downsampling con funzione bicubic per creare immagini LR.
    try:
        train_dataset = SRDataset(
            hr_dir='SR/ARSGN/datasets/test',
            lr_downsampling_method='bicubic'
        )
    except NameError:
        print("ERRORE: La classe SRDataset non è definita. Assicurati che 'dataset.py' esista nella cartella ARSGN.")
        return

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TOTAL_BATCH_SIZE,
        shuffle=True,
        num_workers=8 # Imposta in base alle risorse del tuo sistema
    )

    # --- 2. Inizializzazione Modello e Loss ---
    try:
        # Crea un'istanza del modello ARSGN
        model = Net(args).to(device) # <<<<<< AGGIORNATO: ARSGN
    except NameError:
        print("ERRORE: La classe ARSGN non è definita. Assicurati che 'model.py' esista nella cartella ARSGN.")
        return

    # Se si utilizzano più GPU, avvolgi il modello in DataParallel
    if torch.cuda.device_count() > 1 and N_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))

    # Uso L1Loss (MAE) o L2Loss (MSE). Se il modello è G-N (Generative Network)
    # in un GAN, potrebbe essere una combinazione di L1/L2 e Loss Avversariale.
    # Per una sola subnetwork, L1 è comune.
    criterion = nn.L1Loss().to(device)

    # --- 3. Ottimizzatore ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=INITIAL_LR,
        betas=(BETA1, BETA2),
        eps=EPSILON
    )

    # --- 4. Loop di Training ---
    print("Inizio Training ARSGN...") # <<<<<< AGGIORNATO: ARSGN
    for epoch in range(1, MAX_EPOCHS + 1):
        # Aggiornamento del Learning Rate
        current_lr = adjust_learning_rate(optimizer, epoch)
        print(f"Epoca [{epoch}/{MAX_EPOCHS}], Learning Rate: {current_lr:.1e}")

        model.train()
        for batch_idx, (lr_input, hr_target) in enumerate(train_loader):
            # Sposta i dati sulla GPU
            lr_input = lr_input.to(device)
            hr_target = hr_target.to(device)

            # Forward pass
            sr_output = model(lr_input)

            # Calcolo Loss
            loss = criterion(sr_output, hr_target)

            # Backward e Ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Salvataggio Checkpoint (Esempio)
        if epoch % 100 == 0:
            save_path = f'./ARSGN_checkpoint_epoch_{epoch}.pth' # <<<<<< AGGIORNATO: ARSGN
            print(f"Salvataggio checkpoint in {save_path}")
            # Salva solo lo state_dict del modello non avvolto da DataParallel
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)

    print("Training ARSGN terminato.") # <<<<<< AGGIORNATO: ARSGN

if __name__ == '__main__':
    main()