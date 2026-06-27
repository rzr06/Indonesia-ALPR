"""
Module for YOLO-based Oriented Bounding Box (OBB) detection.
"""
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    """
    A wrapper class for YOLO object detection models.
    
    Attributes:
        model (YOLO): The loaded YOLO model from ultralytics.
        device (str): The computation device used for inference (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, model_path, device="cpu"):
        """
        Initializes the YOLODetector with the specified model weights.

        Args:
            model_path (str): The file path to the YOLO model weights (.pt).
            device (str, optional): The device to run inference on. Defaults to "cpu".
        """
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image):
        """
        Performs object detection on a given image using the YOLO model.

        Args:
            image (numpy.ndarray): The input image array (BGR format typical for OpenCV).

        Returns:
            list: A list of dictionaries containing detection details. Each dict has:
                - "points": numpy.ndarray of the oriented bounding box coordinates.
                - "cls": int, the class ID of the detected object.
                - "conf": float, the confidence score of the detection.
        """
        # Melakukan prediksi pada gambar input dengan ukuran 320x320 dan threshold confidence 0.6
        # # TODO: [Saran perbaikan] Hardcode imgsz=320 dan conf=0.6 dapat dipindahkan ke argumen fungsi atau inisialisasi class agar lebih dinamis.
        results = self.model.predict(
            source=image,
            imgsz=320,
            conf=0.6,
            device=self.device,
            verbose=False
        )

        detections = []
        for r in results:
            if r.obb is None:
                continue

            # Mengekstrak informasi bounding box berorientasi, kelas, dan tingkat confidence
            for box, cls, conf in zip(r.obb.xyxyxyxy, r.obb.cls, r.obb.conf):
                detections.append({
                    "points": box.cpu().numpy(),
                    "cls": int(cls),
                    "conf": float(conf)
                })

        return detections
