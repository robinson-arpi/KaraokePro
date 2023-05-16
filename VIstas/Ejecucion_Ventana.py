import qdarkstyle
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from VentanaPrincipal import Ui_MainWindow

app = QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet_pyside2())  # Agregar esta línea para aplicar el estilo

window = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window)

window.show()
sys.exit(app.exec_())
