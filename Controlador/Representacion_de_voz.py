import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def representacion_audio(audio_file, audio_file2, start_time, end_time):
    audio, sr = librosa.load(audio_file)
    audio2, sr2 = librosa.load(audio_file2)

    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    y_harmonic2, y_percussive2 = librosa.effects.hpss(audio2)

    # Seleccionar solo los primeros 5 segundos
    audio_length = audio.shape[0] / sr
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_harmonic_5s = y_harmonic[start_sample:end_sample]
    y_harmonic2_5s = y_harmonic2[start_sample:end_sample]

    # Graficar las señales de los audios en el mismo gráfico
    plt.figure(figsize=(18, 4))
    librosa.display.waveshow(y_harmonic_5s, sr=sr, alpha=0.5, color='blue', label = 'Prueba')
    librosa.display.waveshow(y_harmonic2_5s, sr=sr2, alpha=0.5, color='red', label = 'Original')
    plt.xlabel("Tiempo (segundos)")
    plt.ylabel("Amplitud")
    titulo = "Gráfico de la señal de la componente armónica (voz) en el intervalo [" + str(start_time) + ", " + str(end_time) + "] segundos"
    plt.title(titulo)
    plt.ylim((-1, 1))

    # Guardar el gráfico como imagen
    nombre_archivo = 'entonacion.png'
    plt.savefig(nombre_archivo)
    print(f'Imagen guardada como {nombre_archivo}.')

