import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Ruta del archivo de audio en formato .mp3
archivo_audio = 'Me voy a olvidar_robinson.mp3'
# Cargar el archivo de audio utilizando librosa
audio, sr = librosa.load(archivo_audio)


# Extraer las características de tono utilizando librosa
tono, _ = librosa.piptrack(y=audio, sr=sr)

# Obtener el promedio del tono a lo largo de las frecuencias
promedio_tono = tono.mean(axis=0)

# Configurar la figura y los ejes del gráfico
plt.figure(figsize=(10, 6))
plt.title('Gráfico de Entonación')
plt.xlabel('Tiempo')
plt.ylabel('Frecuencia')

# Ajustar las dimensiones de promedio_tono para que coincida con el tamaño del gráfico
promedio_tono = promedio_tono[:len(promedio_tono)//2, :]  # Reducir a la mitad el tamaño vertical

# Generar el gráfico de entonación
librosa.display.specshow(promedio_tono, y_axis='log', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')

# Mostrar el gráfico
plt.show()