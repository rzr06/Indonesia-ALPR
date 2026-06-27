"""
Entry point for the Automatic License Plate Recognition application.
"""
import sys
import os
from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow
from core.engine import ALPREngine

def resource_path(relative_path):
    """
    Get the absolute path to a resource. This is required for PyInstaller
    to find files when the app is bundled into an executable.

    Args:
        relative_path (str): The relative path to the resource.

    Returns:
        str: The absolute path to the resource.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Konstanta path untuk model
YOLO_MODEL_PATH = resource_path("models/yolo_obb.pt")
CRNN_MODEL_PATH = resource_path("models/crnn.pth")
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    print("Memuat model YOLO dan CRNN...")
    # Menginisialisasi ALPREngine
    # TODO: [Saran perbaikan] Tangkap kemungkinan exception jika file model tidak ditemukan
    engine = ALPREngine(
        yolo_path=YOLO_MODEL_PATH, 
        crnn_path=CRNN_MODEL_PATH, 
        charset=CHARSET
    )
    
    # Menampilkan Jendela Utama UI
    window = MainWindow(engine)
    window.show()
    
    # Menjalankan event loop aplikasi
    sys.exit(app.exec_())