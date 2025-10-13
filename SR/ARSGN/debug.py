import torch

MODEL_PATH = 'checkpoints/simulated_weights.pth' # IL PERCORSO CORRETTO DEL TUO FILE

try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu') # Carica il file in memoria
    
    print(f"File di checkpoint caricato con successo. Contiene le seguenti chiavi principali:")
    print(checkpoint.keys()) # Stampa tutte le chiavi del dizionario principale
    
    # Prova a identificare la chiave che contiene i pesi
    model_key = None
    for key in checkpoint.keys():
        # Lo stato del modello è generalmente un dizionario di tensori
        if isinstance(checkpoint[key], dict) and any(name.endswith('.weight') for name in checkpoint[key].keys()):
            model_key = key
            break
            
    if model_key:
        print(f"\nLa chiave più probabile che contiene i pesi del modello è: '{model_key}'")
        # Stampa un esempio di un peso per verifica
        first_weight_name = next(iter(checkpoint[model_key].keys()))
        print(f"Esempio di peso: {first_weight_name} con dimensione {checkpoint[model_key][first_weight_name].size()}")
    else:
        print("\nATTENZIONE: Nessuna chiave è stata identificata come stato del modello. Il file potrebbe essere danneggiato o avere un formato non convenzionale.")

except Exception as e:
    print(f"Errore nel caricamento del checkpoint: {e}")