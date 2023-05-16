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
ruta_canciones = ruta_absoluta + '\\Canciones'

# Obtener los nombres de los archivos en la carpeta actual que tienen la extensión .mp3
nombres_canciones = [file_name for file_name in os.listdir(ruta_canciones) if (file_name.endswith(".mp3") or
                                                                               file_name.endswith(".wav"))]


# Creamos una lista de tensores correspondientes a las canciones para poder formar el conjunto de entrenamiento
tensores = []
etiquetas = []

for cancion in nombres_canciones:
    # Aquí sacamos la canción original y le asignamos la misma etiqueta
    audio, sr = librosa.load(ruta_canciones + "\\" + cancion)
    caracteristicas = librosa.feature.melspectrogram(y=audio, sr=sr)
    tensor = torch.from_numpy(caracteristicas).float()
    tensores.append(tensor)

    if cancion.endswith(".mp3"):
        etiquetas.append(cancion[:-4])
    else:
        etiquetas.append(cancion[:-8])
etiquetas_numericas = []

#
for i in range(len(etiquetas)):
    if etiquetas[i] == "Desde Lejos": etiquetas_numericas.append(0)
    if etiquetas[i] == "La Playa": etiquetas_numericas.append(1)
    if etiquetas[i] == "Lo noto": etiquetas_numericas.append(2)
    if etiquetas[i] == "Me voy a olvidar": etiquetas_numericas.append(3)
    if etiquetas[i] == "Por las noches": etiquetas_numericas.append(4)
    if etiquetas[i] == "Te conozco": etiquetas_numericas.append(5)


dataset = list(zip(tensores, etiquetas_numericas))


trainloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)


model = nn.Sequential(nn.Linear(128*10338, 128),
                      nn.ReLU(),
                      nn.Linear(128, 80),
                      nn.ReLU(),
                      nn.Linear(80, 64),
                      nn.ReLU(),
                      nn.Linear(64, 6),
                      nn.LogSoftmax(dim=1))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.RMSprop(model.parameters(), lr=0.001)

epochs = 20
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


# Almacenar el modelo
# Especifica la ruta y el nombre de archivo para guardar el modelo
ruta_archivo = ruta_absoluta + "\\modelo.pth"
# Guarda el estado del modelo y otros elementos necesarios
checkpoint = {
    "modelo": model.state_dict(),
    "optimizador": optimizer.state_dict(),
    "epoch": epochs,
    "criterion": criterion.state_dict()
}
# Guarda el checkpoint en el archivo
torch.save(checkpoint, ruta_archivo)
