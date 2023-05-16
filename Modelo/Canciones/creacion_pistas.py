import os
import librosa
import soundfile as sf
import numpy as np
from scipy import signal

# archivo llega con ruta absoluta
def generar_reverberacion(audio, sample_rate, nombre, decay_time, delay, wet_gain):
    # Generar la respuesta al impulso de la reverberación
    impulse_response = np.zeros(int(decay_time * sample_rate))
    impulse_response[0] = 1.0
    impulse_response = signal.lfilter([1], [1, 0], impulse_response)
    impulse_response = np.concatenate(([1], np.zeros(int(delay * sample_rate)), impulse_response))

    # Ajustar la longitud del impulso de respuesta si es menor que la del audio original
    if len(impulse_response) < len(audio):
        impulse_response = np.pad(impulse_response, (0, len(audio) - len(impulse_response)), 'constant')

    # Aplicar reverberación al audio
    reverberated_audio = signal.convolve(audio, impulse_response, mode='full')

    # Ajustar la longitud del audio reverberado si es mayor que la del audio original
    if len(reverberated_audio) > len(audio):
        reverberated_audio = reverberated_audio[:len(audio)]

    # Aplicar ganancia al audio reverberado
    reverberated_audio = reverberated_audio * (10 ** (wet_gain / 20))

    # Sumar el audio reverberado al audio original
    audio_con_reverberacion = audio + reverberated_audio

    # Normalizar el audio para evitar distorsiones
    audio_con_reverberacion = librosa.util.normalize(audio_con_reverberacion)

    # Guardar el audio con reverberación en formato WAV
    sf.write(nombre[:-4] + "_rev.wav", audio_con_reverberacion, sample_rate)



def generar_eco(audio, sample_rate, nombre, retardo, factor_retardo, nivel_eco):
    # Calcular el número de muestras de retardo
    muestras_retardo = int(retardo * sample_rate)

    # Ajustar el factor de retardo
    factor_retardo = min(1.0, factor_retardo)

    # Obtener el segmento del audio que se utilizará para generar el eco
    eco_segmento = audio[:int(len(audio) * factor_retardo)]

    # Agregar ceros al principio del segmento del eco para crear el retardo
    eco_retardado = np.concatenate((np.zeros(muestras_retardo), eco_segmento))

    # Ajustar el nivel del eco
    eco_retardado = eco_retardado * nivel_eco

    # Verificar si el eco retardado es más corto que el audio original y ajustar su longitud
    if len(eco_retardado) < len(audio):
        eco_retardado = np.pad(eco_retardado, (0, len(audio) - len(eco_retardado)), 'constant')

    # Sumar el eco al audio original
    audio_con_eco = audio + eco_retardado

    # Normalizar el audio para evitar distorsiones
    audio_con_eco = audio_con_eco / np.max(np.abs(audio_con_eco))

    # Guardar el audio con eco en formato WAV
    sf.write(nombre[:-4] + "_eco.wav", audio_con_eco, sample_rate)



def generar_ruido(audio, sample_rate, nombre, nivel_ruido):
    # Generar ruido con la misma longitud que el audio
    ruido = np.random.normal(0, 1, len(audio))

    # Ajustar el nivel del ruido
    ruido_ajustado = ruido * nivel_ruido

    # Sumar el ruido al audio original
    audio_con_ruido = audio + ruido_ajustado

    # Normalizar el audio para evitar distorsiones
    audio_con_ruido = audio_con_ruido / np.max(np.abs(audio_con_ruido))

    # Guardar el audio con ruido en formato WAV
    sf.write(nombre[:-4] + "_rui.wav", audio_con_ruido, sample_rate)    

def cambiar_tono(audio, sample_rate, nombre, semitonos):
    # Calcular el factor de cambio de tono
    factor_tono = 2 ** (semitonos / 12)

    # Aplicar el cambio de tono al audio
    audio_cambiado_tono = librosa.effects.pitch_shift(audio, sample_rate, n_steps=semitonos)

    # Ajustar la longitud del audio cambiado de tono si es mayor que el audio original
    if len(audio_cambiado_tono) > len(audio):
        audio_cambiado_tono = audio_cambiado_tono[:len(audio)]
    else:
        audio_cambiado_tono = np.pad(audio_cambiado_tono, (0, len(audio) - len(audio_cambiado_tono)))

    # Normalizar el audio para evitar distorsiones
    audio_cambiado_tono = librosa.util.normalize(audio_cambiado_tono)

    # Guardar el audio con el tono cambiado en formato WAV
    sf.write(nombre[:-4] + "_ton.wav", audio_cambiado_tono, sample_rate)



# Obtener la ruta absoluta de la carpeta actual
ruta_absoluta = os.path.dirname(os.path.abspath(__file__))
print("Ruta: " + str(ruta_absoluta))

# Obtener los nombres de los archivos en la carpeta actual que tienen la extensión .mp3
nombres_canciones = [file_name for file_name in os.listdir(ruta_absoluta) if file_name.endswith(".mp3")]

# Imprimir los nombres de los archivos
for cancion in nombres_canciones:
    print(cancion)

for cancion in nombres_canciones:
    nombre = ruta_absoluta + "/" + cancion
    audio, sample_rate = librosa.load(nombre)

    generar_reverberacion(audio, sample_rate,nombre, 1, 0.2, -8)
    generar_eco(audio, sample_rate, nombre, 0.5, 0.2, 0.4)
    generar_ruido(audio, sample_rate, nombre, .02)
    #cambiar_tono(audio, sample_rate, nombre, 2)
    