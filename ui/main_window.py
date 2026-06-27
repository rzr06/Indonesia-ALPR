"""
Module containing the main graphical user interface window for the ALPR system.
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QFrame, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

from ui.video_thread import VideoThread

class MainWindow(QMainWindow):
    """
    The main window class for the ALPR application interface.
    
    Attributes:
        engine (ALPREngine): The ALPR engine handling detection and OCR.
        thread (VideoThread): Thread for processing video feeds without freezing the UI.
    """

    def __init__(self, engine):
        """
        Initializes the MainWindow.

        Args:
            engine (ALPREngine): An instance of the initialized ALPREngine.
        """
        super().__init__()
        self.engine = engine
        self.setWindowTitle("ALPR Dashboard - Toll Road Monitoring System")
        self.setGeometry(100, 100, 1280, 720)
        self.thread = None

        self.apply_dark_theme()
        self.init_ui()

    def apply_dark_theme(self):
        """
        Applies a modern dark theme to the application components using QSS.
        """
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #E0E0E0; font-family: 'Segoe UI', Arial; }
            QFrame#Panel { background-color: #1E1E1E; border-radius: 8px; }
            
            QPushButton {
                background-color: #2D2D30; color: #FFFFFF;
                border: 1px solid #3E3E42; border-radius: 5px;
                padding: 10px; font-weight: bold; font-family: 'Segoe UI';
            }
            QPushButton:hover { background-color: #3E3E42; border: 1px solid #007ACC; }
            QPushButton:pressed { background-color: #007ACC; }
            QPushButton#StopBtn { background-color: #8A2BE2; border: 1px solid #9932CC; }
            QPushButton#StopBtn:hover { background-color: #9400D3; }
            
            QTableWidget {
                background-color: #1E1E1E; color: #00FF00;
                gridline-color: #333333; border: 1px solid #333333;
                border-radius: 5px; font-family: 'Consolas', monospace; font-size: 14px;
            }
            QHeaderView::section {
                background-color: #2D2D30; color: #E0E0E0;
                padding: 5px; border: 1px solid #333333; font-weight: bold;
            }
        """)

    def init_ui(self):
        """
        Initializes and arranges the UI layouts, buttons, panels, and tables.
        """
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("INTELLIGENT TRAFFIC MONITORING - ALPR SYSTEM")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setStyleSheet("color: #007ACC; letter-spacing: 2px;")
        
        self.time_label = QLabel("00:00:00")
        self.time_label.setFont(QFont("Consolas", 16, QFont.Bold))
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.time_label)
        main_layout.addLayout(header_layout)

        # Realtime clock timer
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        # Content Area
        content_layout = QHBoxLayout()
        
        # Panel Kiri: Video Feed
        video_panel = QFrame()
        video_panel.setObjectName("Panel")
        video_layout = QVBoxLayout(video_panel)
        
        cam_info_layout = QHBoxLayout()
        cam_title = QLabel("LIVE CAMERA FEED")
        cam_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.fps_label = QLabel("FPS: 0.00")
        self.fps_label.setFont(QFont("Consolas", 10))
        self.fps_label.setStyleSheet("color: #00FF00;")
        cam_info_layout.addWidget(cam_title)
        cam_info_layout.addStretch()
        cam_info_layout.addWidget(self.fps_label)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 480)
        self.image_label.setStyleSheet("background-color: #000000; border: 2px solid #333333; border-radius: 4px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("NO SIGNAL / WAITING FOR INPUT")
        self.image_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        
        video_layout.addLayout(cam_info_layout)
        video_layout.addWidget(self.image_label, stretch=1)
        content_layout.addWidget(video_panel, stretch=7)

        # Panel Kanan: Kontrol dan Log Deteksi
        right_panel = QVBoxLayout()
        
        # Kontrol Panel
        control_frame = QFrame()
        control_frame.setObjectName("Panel")
        control_layout = QVBoxLayout(control_frame)
        control_layout.setSpacing(10)
        
        ctrl_title = QLabel("SYSTEM CONTROLS")
        ctrl_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        control_layout.addWidget(ctrl_title)

        self.btn_image = QPushButton("📸 Upload Image")
        self.btn_video = QPushButton("🎥 Upload Video")
        self.btn_camera = QPushButton("🔴 Start Live Camera")
        self.btn_stop = QPushButton("⏹ Stop Processing")
        self.btn_stop.setObjectName("StopBtn")

        control_layout.addWidget(self.btn_image)
        control_layout.addWidget(self.btn_video)
        control_layout.addWidget(self.btn_camera)
        control_layout.addWidget(self.btn_stop)
        
        # Log Table Panel
        data_frame = QFrame()
        data_frame.setObjectName("Panel")
        data_layout = QVBoxLayout(data_frame)
        
        data_title = QLabel("DETECTION LOGS")
        data_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        
        self.table_widget = QTableWidget(0, 3)
        self.table_widget.setHorizontalHeaderLabels(["Time", "Plate Number", "Conf"])
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setStyleSheet("alternate-background-color: #252526;")

        data_layout.addWidget(data_title)
        data_layout.addWidget(self.table_widget)

        right_panel.addWidget(control_frame)
        right_panel.addWidget(data_frame, stretch=1)
        
        content_layout.addLayout(right_panel, stretch=3)
        main_layout.addLayout(content_layout)

        # Event connections
        self.btn_image.clicked.connect(self.load_image)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_processing)

    def update_clock(self):
        """
        Updates the UI clock with the current system time.
        """
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.time_label.setText(current_time)

    def load_image(self):
        """
        Opens a file dialog for the user to select an image, processes it, and updates the UI.
        """
        self.stop_processing()
        filename, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.engine.reset_state()
            frame = cv2.imread(filename)
            processed_frame, results = self.engine.process_frame(frame, is_video=False)
            self.update_image(processed_frame)
            self.update_log(results)

    def load_video(self):
        """
        Opens a file dialog for the user to select a video file and starts the video thread.
        """
        self.stop_processing()
        filename, _ = QFileDialog.getOpenFileName(self, "Pilih Video", "", "Video Files (*.mp4 *.avi)")
        if filename:
            self.start_thread(filename)

    def start_camera(self):
        """
        Starts the video thread capturing from the primary webcam (index 0).
        """
        self.stop_processing()
        self.start_thread(0)

    def start_thread(self, source):
        """
        Starts the VideoThread with the given source (file path or camera index).

        Args:
            source (str or int): Video source to process.
        """
        self.thread = VideoThread(self.engine)
        self.thread.set_source(source)
        # Menghubungkan signal dari thread ke slot di UI
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_log_signal.connect(self.update_log)
        self.thread.update_fps_signal.connect(self.update_fps)
        self.thread.start()

    def stop_processing(self):
        """
        Stops the video processing thread safely and resets the UI state.
        """
        if self.thread is not None:
            try:
                self.thread.change_pixmap_signal.disconnect()
                self.thread.update_log_signal.disconnect()
                self.thread.update_fps_signal.disconnect()
            except TypeError:
                pass
                
            self.thread.stop()
            self.thread = None
            
        self.engine.reset_state()
        self.fps_label.setText("FPS: 0.00")
        self.image_label.clear()
        self.image_label.setText("NO SIGNAL / WAITING FOR INPUT")
        self.image_label.setStyleSheet("background-color: #000000; color: #E0E0E0; border: 2px solid #333333; border-radius: 4px;")
        self.image_label.setAlignment(Qt.AlignCenter)

    def update_image(self, cv_img):
        """
        Slot to receive OpenCV image arrays and display them on the image_label.

        Args:
            cv_img (numpy.ndarray): The processed BGR image from OpenCV.
        """
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def update_fps(self, fps):
        """
        Slot to update the FPS counter label.

        Args:
            fps (float): Frames per second.
        """
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def update_log(self, results):
        """
        Slot to update the detection log table with newly recognized plates.

        Args:
            results (list): List of dictionaries containing "text" and "conf".
        """
        current_time = QDateTime.currentDateTime().toString("HH:mm:ss")
        for res in results:
            self.table_widget.insertRow(0)
            
            self.table_widget.setItem(0, 0, QTableWidgetItem(current_time))
            
            plate_item = QTableWidgetItem(res['text'])
            plate_item.setFont(QFont("Consolas", 12, QFont.Bold))
            self.table_widget.setItem(0, 1, plate_item)
            
            conf_item = QTableWidgetItem(f"{res['conf']:.2f}")
            if res['conf'] < 0.7:
                conf_item.setForeground(Qt.red)
            self.table_widget.setItem(0, 2, conf_item)

            # Menjaga agar log tabel tidak memori-intensive dengan limit 100 baris
            if self.table_widget.rowCount() > 100:
                self.table_widget.removeRow(100)

    def convert_cv_qt(self, cv_img):
        """
        Converts an OpenCV BGR image into a QPixmap suitable for Qt rendering.

        Args:
            cv_img (numpy.ndarray): The OpenCV BGR image.

        Returns:
            QPixmap: The converted Qt pixmap image.
        """
        # Konversi color space OpenCV BGR ke RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        label_size = self.image_label.size()
        p = convert_to_Qt_format.scaled(label_size.width(), label_size.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(p)