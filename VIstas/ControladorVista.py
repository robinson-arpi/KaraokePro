from mutagen import File

# Ruta al archivo de audio
ruta_archivo = "ClasificacionDeCanciones\Modelo\Canciones\Por las noches.mp3"

# Cargar el archivo de audio
audio = File(ruta_archivo)

# Obtener los metadatos
artist = audio.tags.get("artist")
title = audio.tags.get("title")

# Imprimir los metadatos
print("Artista:", artist)
print("Título:", title)
