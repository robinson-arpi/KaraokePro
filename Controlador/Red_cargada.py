import os
import torch
import librosa
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

class Red():
    def __init__(self):
        self.modelo = None
        self.ruta = None
        self.cargar_ruta()
        self.cargar_modelo()
        self.cargar_checkpoint()

    def cargar_ruta(self):
        # Obtener la ruta absoluta de la carpeta actual
        self.ruta = os.path.dirname(os.path.abspath(__file__))
        print("RUTA: " + str(self.ruta)[:-11] + "\n")

    def cargar_modelo(self):
        # Crea una nueva instancia del modelo
        self.modelo = nn.Sequential(nn.Linear(128*10338, 128),
                      nn.ReLU(),
                      nn.Linear(128, 80),
                      nn.ReLU(),
                      nn.Linear(80, 64),
                      nn.ReLU(),
                      nn.Linear(64, 6),
                      nn.LogSoftmax(dim=1))

    def cargar_checkpoint(self):
        # Restaura el estado del modelo y otros elementos desde el checkpoint
        checkpoint = torch.load(self.ruta[:-11] + "\\Modelo\\modelo.pth")
        self.modelo.load_state_dict(checkpoint["modelo"])
        criterion = nn.NLLLoss()
        criterion.load_state_dict(checkpoint["criterion"])
        optimizer = optim.SGD(self.modelo.parameters(), lr=0.0001)
        optimizer.load_state_dict(checkpoint["optimizador"])
        epochs = checkpoint["epoch"]
    

    def nueva_probabilidad(self):
        audio_prueba, sr_prueba = librosa.load(self.ruta)
        caracteristicas_prueba = librosa.feature.melspectrogram(y=audio_prueba, sr=sr_prueba)
        tensor_prueba = torch.from_numpy(caracteristicas_prueba).float()
        tensor_prueba = tensor_prueba.view(1,-1)
        output_prueba = self.modelo(tensor_prueba)
        probabilidades_prueba = F.softmax(output_prueba, dim=1)
        etiqueta_predicha = torch.argmax(probabilidades_prueba, dim=1).item()

        print(etiqueta_predicha)

general = []
def evaluar_cancion(ruta):
    red_neuronal = Red()

    audio, sr = librosa.load(ruta)
    
    #ajustar
    target_length = 5292864

    if len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    elif len(audio) > target_length:
        audio = audio[:target_length]

    
    caracteristicas = librosa.feature.melspectrogram(y=audio, sr=sr)
    tensor_prueba = torch.from_numpy(caracteristicas).float()
    tensor_prueba = tensor_prueba.view(1,-1)
    logits = red_neuronal.modelo.forward(tensor_prueba)

    # Apply softmax to the logits to get probabilities
    ps = F.softmax(logits, dim=1)
    print(ps)
    lista = ps.tolist()[0]
    '''
    probs = ps.squeeze().tolist()
    maximo = max(probs)
    probs[probs.index(maximo)] = 2-sum(probs)
    print(probs)
    
    
    # Calculate the percentage values
    percentages = ["{:0.2e}".format(p * 100)  if p < 0.0001 else (p*100) for p in probs]
    print(percentages)
    '''
    suma_total = sum(lista)

    percentages = [round(valor * 100 / suma_total, 2) for valor in lista]
    # Create a bar chart of the probabilities
    fig, ax = plt.subplots()
    ax.bar(range(6), lista)
    ax.set_xticks(range(6))
    ax.set_xticklabels(['Desde Lejos', 'La Playa', 'Lo noto', 'Me voy','Por las noches', 'Te conozco'])
    ax.set_ylabel('Probability')
    ax.set_title('Classification Probabilities')
    # Add percentage labels to the bars
    for i, p in enumerate(percentages):
        ax.text(i, lista[i] + 0.01, str(p) + '%', ha='center')
    # Guardar el gráfico como imagen
    nombre_archivo = 'barras.png'
    plt.savefig(nombre_archivo)
    print(f'Imagen guardada como {nombre_archivo}.')
    return percentages


