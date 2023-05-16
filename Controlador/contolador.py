from mutagen.mp3 import MP3
import librosa
import os
import sys
import qdarkstyle
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene
from VentanaPrincipal import Ui_MainWindow
from pydub import AudioSegment

import threading
import tkinter as tk
from tkinter import filedialog
import Representacion_de_voz as rv
from PyQt5.QtGui import QPixmap
import pygame
import sounddevice as sd
import soundfile as sf
import Red_cargada as rc
import matplotlib.pyplot as plt

class MiVentana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ruta = ""
        self.ruta_archivo_cargado = ""
        #self.cargar_modelo()
        self.titulo = None
        self.ui.tabResultados.setCurrentIndex(0)
        self.mostrar_grafico_clasificacion()
        self.reproduccion_en_progreso = False  # Variable para controlar la reproducción
        self.hilo_reproduccion = None  # Variable para almacenar el hilo de reproducción
        self.hilo_entonacion = None  # Variable para almacenar el hilo de reproducción
        self.hilo_grabacion = None
        
        self.cargar_canciones()
        self.ui.btnSeleccionar.clicked.connect(self.seleccionar_informacion_cancion)
        self.ui.btnGrabar.clicked.connect(self.reproducir)
        self.ui.btnCancelar.clicked.connect(self.detener_reproduccion)
        self.ui.btnCargar.clicked.connect(self.seleccionar_archivo_audio)
        self.ui.btnSeleccionarTramo.clicked.connect(self.mostrar_grafico_entonacion)
    
    def cargar_canciones(self):
        matriz_datos = [ ["Desde Lejos", "Santiago Cruz", "4:00"],
                    ["La Playa", "La Oreja de Van Goh", "4:00"],
                    ["Lo noto", "Hombres G", "4:00"],
                    ["Me voy a olvidar", "T&K", "4:00"],
                    ["Por las noches", "Peso Pluma", "4:00"],
                    ["Te conozco", "Ricardo Arjona", "4:00"],
                    ]
        
        # Obtener la cantidad de filas y columnas de la matriz de datos
        num_filas = len(matriz_datos)
        num_columnas = len(matriz_datos[0])


        # Establecer el número de filas y columnas del QTableWidget
        self.ui.tblCanciones.setRowCount(num_filas)
        self.ui.tblCanciones.setColumnCount(num_columnas)
        

        # Iterar sobre la matriz de datos y asignar los valores a las celdas del QTableWidget
        for fila, datos_fila in enumerate(matriz_datos):
            for columna, dato in enumerate(datos_fila):
                item = QtWidgets.QTableWidgetItem(str(dato))
                self.ui.tblCanciones.setItem(fila, columna, item)
        # Establecer el ancho predeterminado de las columnas
        self.ui.tblCanciones.setColumnWidth(0, 250)  # Columna 0 con ancho de 200 píxeles
        self.ui.tblCanciones.setColumnWidth(1, 250)  # Columna 1 con ancho de 250 píxeles
        self.ui.tblCanciones.setColumnWidth(2, 95)  # Columna 2 con ancho de 100 píxeles
        
    
        #self.ui.tblCanciones.resizeColumnsToContents()
    
    def seleccionar_informacion_cancion(self):
        fila_seleccionada = self.ui.tblCanciones.currentRow()
        
        if fila_seleccionada >= 0:
            nombre_cancion = self.ui.tblCanciones.item(fila_seleccionada, 0).text()
            artista = self.ui.tblCanciones.item(fila_seleccionada, 1).text()
            duracion = self.ui.tblCanciones.item(fila_seleccionada, 2).text()
            
            informacion_cancion = f"Canción: {nombre_cancion}\nArtista: {artista}\nDuración: {duracion}"
            self.ruta = self.obtener_ruta() + "\\"+ nombre_cancion + ".mp3"
            self.ui.lblCancion.setText(informacion_cancion)
            self.titulo = nombre_cancion + ".mp3"
        else:
            self.ui.lblCancion.setText("Ninguna canción seleccionada")
    
    def capturar_audio(self, duracion, tasa_bits):
        fs = 44100  # Frecuencia de muestreo
        canales = 2  # Número de canales de audio (estéreo)

        # Configurar los parámetros para la grabación
        sd.default.samplerate = fs
        sd.default.channels = canales
        sd.default.dtype = 'int16'

        print("Capturando audio...")
        grabacion = sd.rec(int(duracion * fs), samplerate=fs, channels=canales)
        sd.wait()
        print("Audio capturado.")

        # Guardar la grabación en un archivo WAV con la calidad de codificación especificada
        nombre_archivo = "grabacion.wav"
        sf.write(nombre_archivo, grabacion, fs, format='WAV', subtype='PCM_16')

        print(f"Grabación guardada en {nombre_archivo}.")
        self.cargar_tab2()

    def grabar(self):    
        # Duración y tasa de bits deseadas
        duracion_minutos = 4
        tasa_bits_kbps = 128

        # Convertir duración a segundos
        duracion_segundos = duracion_minutos * 60
        
        # Realizar la captura de audio
        self.capturar_audio(duracion_segundos, tasa_bits_kbps * 1000)

    def reproducir_cancion(self):
        self.reproduccion_en_progreso = True  # Indicar que la reproducción está en progreso
        # Ruta de la canción a reproducir
        ruta_cancion = self.ruta
        pygame.mixer.init()
        pygame.mixer.music.load(ruta_cancion)
        pygame.mixer.music.play()
        self.reproduccion_en_progreso = False  # Indicar que la reproducción ha finalizado
        
    def reproducir(self):
        if not self.reproduccion_en_progreso:
            self.ui.btnGrabar.setStyleSheet("background-color: rgba(255, 0, 0, 100);")
            self.hilo_reproduccion = threading.Thread(target=self.reproducir_cancion)
            self.hilo_reproduccion.start()
            self.hilo_grabacion = threading.Thread(target = self.grabar)
            self.hilo_grabacion.start()


    def detener_reproduccion(self):
        if self.reproduccion_en_progreso:
            self.hilo_reproduccion.join()
            self.reproduccion_en_progreso = False
            self.restaurar_color_boton_grabar()  # Restaurar el color original del botón

    def restaurar_color_boton_grabar(self):
        self.ui.btnGrabar.setStyleSheet("")  # Restaurar el estilo original del botón    
            
    def obtener_ruta(self):
        # Obtener la ruta absoluta del directorio actual
        directorio_actual = os.getcwd()

        # Obtener la ruta absoluta de la carpeta actual
        ruta = directorio_actual + "\\Pistas"
        return ruta
    

    def seleccionar_archivo_audio(self):
        root = tk.Tk()
        root.withdraw()

        # Abrir el diálogo de selección de archivos
        self.ruta_archivo_cargado = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=[("Archivos de audio", "*.mp3;*.wav"), ("Todos los archivos", "*.*")]
        )

        # Comprobar si se seleccionó un archivo
        if self.ruta_archivo_cargado:
            rc.evaluar_cancion(self.ruta_archivo_cargado)
            self.mostrar_grafico_clasificacion()
            #self.generar_grafico_entonacion2()
            self.ui.tabResultados.setCurrentIndex(1)

    def generar_grafico_entonacion2(self):
        ruta= self.ruta_archivo_cargado
       
        audio, sr = librosa.load(ruta)

        # Extraer el pitch contour
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # Visualizar el pitch contour en un gráfico de entonación en líneas
        plt.figure(figsize=(12, 4))
        plt.plot(f0)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.title('Gráfico de entonación en líneas')
        plt.show()

    def mostrar_grafico_entonacion(self):
        # Generar el gráfico de representación de audio
        ruta_original = 'Modelo\\Canciones\\' + self.titulo
        rv.representacion_audio(str(self.ruta_archivo_cargado),ruta_original,float(self.ui.txtInicio.text()), float(self.ui.txtFin.text()))

        # Crear un pixmap a partir de la imagen
        pixmap = QPixmap("entonacion.png")
        #pixmap.loadFromData(grafico_image.getvalue())

        # Escalar la imagen al tamaño del QGraphicsView
        #pixmap = pixmap.scaled(self.ui.graficoEntonacion.size())

        # Crear una escena y agregar el pixmap
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)

        # Establecer la escena en el QGraphicsView
        self.ui.graficoEntonacion.setScene(scene)

    def mostrar_grafico_clasificacion(self):
        # Cargar la imagen desde el archivo "barras.png"
        pixmap = QPixmap("barras.png")

        # Escalar la imagen al tamaño del QGraphicsView
        #pixmap = pixmap.scaled(self.ui.graficoClasificacion.size())

        # Crear una escena y agregar el pixmap
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)

        # Establecer la escena en el QGraphicsView
        self.ui.graficoClasificacion.setScene(scene)

    def cargar_tab2(self):
        self.ruta_archivo_cargado = "grabacion.wav"
        rc.evaluar_cancion(self.ruta_archivo_cargado)
        self.ui.tabResultados.setCurrentIndex(1)
        self.mostrar_grafico_clasificacion()

        

app = QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet_pyside2())  # Agregar esta línea para aplicar el estilo

window = MiVentana()
window.show()
sys.exit(app.exec_())