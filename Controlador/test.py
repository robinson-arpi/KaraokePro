import matplotlib.pyplot as plt
import librosa    

def generar_grafico_entonacion2(inicio, duracion):
    ruta= 'Modelo\\Canciones\La Playa.mp3'
    
    audio, sr = librosa.load(ruta, offset=inicio, duration=duracion)

    # Extraer el pitch contour
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # Configurar el eje x para representar el tiempo en segundos
    tiempo = librosa.times_like(f0, sr=sr)

    # Graficar el contorno de entonación
    plt.figure(figsize=(12, 4))
    plt.plot(tiempo, f0)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.title('Gráfico de Entonación')
    plt.show()

generar_grafico_entonacion2(0.0, 10.0)