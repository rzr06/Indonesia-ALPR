"""
Module providing a worker thread for processing video streams asynchronously.
"""
import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class VideoThread(QThread):
    """
    A QThread subclass that reads frames from a video source or camera
    and processes them using the ALPREngine without blocking the main GUI thread.

    Signals:
        change_pixmap_signal (np.ndarray): Emitted when a frame is processed and ready to be displayed.
        update_log_signal (list): Emitted when one or more license plates are successfully recognized.
        update_fps_signal (float): Emitted with the calculated processing frames per second.
    """
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_log_signal = pyqtSignal(list)
    update_fps_signal = pyqtSignal(float)

    def __init__(self, engine):
        """
        Initializes the VideoThread.

        Args:
            engine (ALPREngine): The inference engine to process frames.
        """
        super().__init__()
        self.engine = engine
        self._run_flag = True
        self.source = 0 

    def set_source(self, source):
        """
        Sets the video source (file path or camera index).

        Args:
            source (str or int): The video source identifier.
        """
        self.source = source

    def run(self):
        """
        Main execution loop for the thread. Captures video frames,
        feeds them to the engine, calculates FPS, and emits signals.
        """
        self.engine.reset_state()
        
        cap = cv2.VideoCapture(self.source)
        
        is_file = isinstance(self.source, str)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Hitung target delay jika memutar file video agar tidak terlalu cepat
        # TODO: [Saran perbaikan] Video loop ini mungkin tidak sempurna sinkron dengan waktu aktual (audio tidak ada).
        target_delay = 1.0 / video_fps if (video_fps > 0 and is_file) else 0

        prev_time = time.time()

        while self._run_flag:
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # Memproses frame menggunakan ALPR Engine
                processed_frame, results = self.engine.process_frame(frame, is_video=True)
                
                elapsed = time.time() - start_time
                if target_delay > 0:
                    sleep_time = target_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                actual_current_time = time.time()
                # Menghindari ZeroDivisionError pada FPS calculation
                fps = 1 / (actual_current_time - prev_time) if (actual_current_time - prev_time) > 0 else 0
                prev_time = actual_current_time
                
                # Mengirim data frame dan FPS ke UI
                self.change_pixmap_signal.emit(processed_frame)
                self.update_fps_signal.emit(fps)
                
                # Jika ada deteksi baru yang terkonfirmasi, log ke UI
                if results:
                    self.update_log_signal.emit(results)
            else:
                break
                
        cap.release()

    def stop(self):
        """
        Signals the thread to stop processing gracefully.
        """
        self._run_flag = False
        self.wait()