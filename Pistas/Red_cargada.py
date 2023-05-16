import os
import torch
import librosa
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

# Obtener la ruta absoluta de la carpeta actual
ruta_absoluta = os.path.dirname(os.path.abspath(__file__))

# Crea una nueva instancia del modelo
modelo_cargado = nn.Sequential(nn.Linear(128*10338, 128),
                      nn.ReLU(),
                      nn.Linear(128, 80),
                      nn.ReLU(),
                      nn.Linear(80, 64),
                      nn.ReLU(),
                      nn.Linear(64, 6),
                      nn.LogSoftmax(dim=1))


# Restaura el estado del modelo y otros elementos desde el checkpoint
checkpoint = torch.load(ruta_absoluta + "\\modelo.pth")
modelo_cargado.load_state_dict(checkpoint["modelo"])

criterion = nn.NLLLoss()
criterion.load_state_dict(checkpoint["criterion"])
optimizer = optim.SGD(modelo_cargado.parameters(), lr=0.0001)
optimizer.load_state_dict(checkpoint["optimizador"])
epochs = checkpoint["epoch"]


def evaluar_cancion(ruta):
    audio, sr = librosa.load(ruta)
    caracteristicas = librosa.feature.melspectrogram(y=audio, sr=sr)
    tensor_prueba = torch.from_numpy(caracteristicas).float()
    tensor_prueba = tensor_prueba.view(1,-1)
    logits = modelo_cargado.forward(tensor_prueba)

    # Apply softmax to the logits to get probabilities
    ps = F.softmax(logits, dim=1)
    probs = ps.squeeze().tolist()
    maximo = max(probs)
    probs[probs.index(maximo)] = 2-sum(probs)
    print(probs)
    
    
    # Calculate the percentage values
    percentages = ["{:0.2e}".format(p * 100)  if p < 0.0001 else (p*100) for p in probs]
    print(percentages)

    # Create a bar chart of the probabilities
    fig, ax = plt.subplots()
    ax.bar(range(6), probs)
    ax.set_xticks(range(6))
    ax.set_xticklabels(['Desde Lejos', 'La Playa', 'Lo noto', 'Me voy a olvidar','Por las noches', 'Te conozco'])
    ax.set_ylabel('Probability')
    ax.set_title('Classification Probabilities')
    # Add percentage labels to the bars
    for i, p in enumerate(percentages):
        ax.text(i, probs[i] + 0.01, str(p) + '%', ha='center')
    plt.show()

def clase_predecida(ruta):
    audio_prueba, sr_prueba = librosa.load(ruta)
    caracteristicas_prueba = librosa.feature.melspectrogram(y=audio_prueba, sr=sr_prueba)
    tensor_prueba = torch.from_numpy(caracteristicas_prueba).float()
    tensor_prueba = tensor_prueba.view(1,-1)
    output_prueba = modelo_cargado(tensor_prueba)
    probabilidades_prueba = F.softmax(output_prueba, dim=1)
    etiqueta_predicha = torch.argmax(probabilidades_prueba, dim=1).item()

    print(etiqueta_predicha)

evaluar_cancion(ruta_absoluta + "\\Testing\\Me voy a olvidar_robinson.mp3" )   
#evaluar_cancion(ruta_absoluta + "\\Testing\\Te conozco_rev.wav" ) 
#evaluar_cancion(ruta_absoluta + "\\Testing\\La Playa_rui.wav" ) 
#evaluar_cancion(ruta_absoluta + "\\Testing\\Lo noto_eco.wav" ) 
evaluar_cancion(ruta_absoluta + "\\Testing\\Por las noches_prueba.mp3")
#evaluar_cancion(ruta_absoluta + "\\Testing\\Desde Lejos_rui.wav")
#evaluar_cancion(ruta_absoluta + "\\Testing\\Por las noches.mp3")