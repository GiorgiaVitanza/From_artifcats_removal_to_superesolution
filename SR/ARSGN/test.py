import torch
from PIL import Image
from torchvision import transforms
import numpy as np
# Assumiamo che la classe Net e i moduli 'common' siano importabili
from agsr import Net
from common import *

# Funzione per caricare e pre-elaborare l'immagine
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # Converti l'immagine in Tensore e normalizza (ad esempio, a [0, 1])
    transform = transforms.ToTensor()
    tensor = transform(img)
    # Aggiungi una dimensione 'Batch' (da C x H x W a 1 x C x H x W)
    return tensor.unsqueeze(0)

# Funzione per salvare l'output
def save_image(tensor_output, output_path):
    # Rimuovi la dimensione Batch (da 1 x C x H x W a C x H x W)
    img_tensor = tensor_output.squeeze(0).cpu() 
    
    # Clampa i valori nell'intervallo valido [0, 1] e converti in numpy
    img_array = img_tensor.clamp(0, 1).numpy()
    
    # Riorganizza i canali (da C x H x W a H x W x C) e scala a [0, 255]
    img_array = np.transpose(img_array, (1, 2, 0)) * 255
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(output_path)


# --- Logica di Esecuzione Principale ---
if __name__ == '__main__':
    # 1. Definisci i percorsi
    MODEL_PATH = 'checkpoints/simulated_weights.pth' # Percorso ai pesi addestrati
    INPUT_IMAGE = 'SR/ARSGN/datasets/test/airplane00.tif' # La tua immagine di test
    OUTPUT_IMAGE = 'results/super_resolved.png'

    # 2. Caricamento del modello
    # Per il tuo modello, devi creare un oggetto 'args' fittizio
    class DummyArgs:
        pass
    args = DummyArgs()
    
    model = Net(args)
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict']) # Carica i pesi
    model.eval() # Modalità di valutazione (disabilita dropout/batchnorm)
    
    # 3. Preparazione dell'Input
    input_tensor = load_image(INPUT_IMAGE)
    
    # 4. Inferenza
    with torch.no_grad(): # Nessun calcolo del gradiente durante il test
        # Il tuo modello restituisce (sr_1, sr_2), prendiamo il risultato finale sr_2
        _, output_tensor = model(input_tensor) 

    # 5. Salvataggio
    save_image(output_tensor, OUTPUT_IMAGE)
    print(f"Immagine salvata in: {OUTPUT_IMAGE}")