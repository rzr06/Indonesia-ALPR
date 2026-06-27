"""
Module acting as the core engine for the Automatic License Plate Recognition system.
It ties together detection, tracking, rectification, and OCR.
"""
import cv2
import numpy as np
import torch
import time
import os
import difflib
from datetime import datetime

from inference.detector import YOLODetector
from utils.geometry import rectify_plate, order_points
from inference.recognizer import CRNNRecognizer
from utils.tracker import CentroidTracker

class ALPREngine:
    """
    Core engine that coordinates YOLO detection, Centroid Tracking, and CRNN OCR.
    
    Attributes:
        device (str): Computation device.
        detector (YOLODetector): The YOLO object detection wrapper.
        recognizer (CRNNRecognizer): The CRNN OCR text recognition wrapper.
        tracker (CentroidTracker): The object tracking algorithm.
    """

    def __init__(self, yolo_path, crnn_path, charset="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "):
        """
        Initializes the ALPREngine with the specified models.

        Args:
            yolo_path (str): File path to the YOLO model weights.
            crnn_path (str): File path to the CRNN model weights.
            charset (str, optional): The recognizable character set.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.detector = YOLODetector(model_path=yolo_path, device=self.device)
        self.recognizer = CRNNRecognizer(model_path=crnn_path, charset=charset)
        self.tracker = CentroidTracker(max_disappeared=10, max_distance=150)
        
        # Mapping kelas deteksi YOLO (disesuaikan dengan hasil training)
        self.nopol_class_id = 0  
        self.plate_class_id = 1  
        
        # Buffer untuk mengumpulkan kandidat bacaan pelat yang sama melalui waktu
        self.track_buffers = {}
        # Histori log teks pelat yang sudah tercatat
        self.logged_texts = {} 
        self.ttl_seconds = 60  
        
        self.log_dir = "logs"
        self.img_log_dir = os.path.join(self.log_dir, "plate_images")
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.img_log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, "plate_history.txt")

    def reset_state(self):
        """
        Resets tracking state and clears internal buffers.
        Useful when switching video streams or processing new standalone images.
        """
        self.tracker = CentroidTracker(max_disappeared=10, max_distance=150)
        self.track_buffers.clear()
        self.logged_texts.clear()

    def log_to_file(self, text, conf):
        """
        Logs the recognized plate text and confidence to a local text file.

        Args:
            text (str): The recognized license plate string.
            conf (float): Confidence score of the detection.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] NOPOL: {text:<12} | CONF: {conf:.4f}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def save_plate_image(self, text, conf, cropped_img):
        """
        Saves the cropped region of the recognized license plate to disk.

        Args:
            text (str): Recognized text (used in filename).
            conf (float): Confidence score (used in filename).
            cropped_img (numpy.ndarray): The rectified image to save.
        """
        time_str = datetime.now().strftime("%H-%M-%S")
        filename = f"{text}_{time_str}_{conf:.4f}.jpg"
        save_path = os.path.join(self.img_log_dir, filename)
        if cropped_img is not None:
            # Menggunakan kompresi JPEG kualitas terbaik
            cv2.imwrite(save_path, cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def clean_expired_plates(self):
        """
        Cleans up the logged texts registry to prevent indefinite memory consumption
        and allow re-reading the same plate if enough time has passed.
        """
        current_time = time.time()
        expired_keys = [k for k, v in self.logged_texts.items() if (current_time - v) > self.ttl_seconds]
        for k in expired_keys:
            del self.logged_texts[k]

    def draw_tilted_text(self, img, text, center_x, center_y, angle_deg, font_scale, thickness, color, outline_color):
        """
        Draws text rotated to match the angle of the oriented bounding box.

        Args:
            img (numpy.ndarray): The image canvas to draw on (modified in-place).
            text (str): The text string to draw.
            center_x (float): The x-coordinate of the text center point.
            center_y (float): The y-coordinate of the text center point.
            angle_deg (float): The rotation angle in degrees.
            font_scale (float): Scaling factor for the font size.
            thickness (int): Thickness of the inner text.
            color (tuple): The BGR color of the text.
            outline_color (tuple): The BGR color of the text outline.
        """
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        pad = int(max(tw, th) * 0.8)
        canvas_w = tw + pad * 2
        canvas_h = th + pad * 2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        cx = canvas_w // 2
        cy = canvas_h // 2
        
        tx = cx - tw // 2
        ty = cy + th // 2
        
        # Membuat outline dan teks bagian dalam
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
        # Rotasi kanvas menggunakan affine transformation
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated_canvas = cv2.warpAffine(canvas, M, (canvas_w, canvas_h))
        
        x_min = int(center_x - cx)
        y_min = int(center_y - cy)
        x_max = x_min + canvas_w
        y_max = y_min + canvas_h
        
        img_h, img_w = img.shape[:2]
        
        x1, x2 = max(0, x_min), min(img_w, x_max)
        y1, y2 = max(0, y_min), min(img_h, y_max)
        
        cx1, cx2 = max(0, -x_min), min(canvas_w, canvas_w - (x_max - img_w))
        cy1, cy2 = max(0, -y_min), min(canvas_h, canvas_h - (y_max - img_h))
        
        # Masking dan blending ke citra asli
        if x1 < x2 and y1 < y2 and cx1 < cx2 and cy1 < cy2:
            mask = np.any(rotated_canvas[cy1:cy2, cx1:cx2] > 0, axis=-1)
            img[y1:y2, x1:x2][mask] = rotated_canvas[cy1:cy2, cx1:cx2][mask]

    def get_best_text_by_weighted_voting(self, readings):
        """
        Determines the most confident reading sequence from a history of tracking attempts.
        It groups similar strings (using difflib) and accumulates their confidence scores.

        Args:
            readings (list): List of dicts containing "text" (str) and "conf" (float).

        Returns:
            tuple: (best_text (str), total_score (float)). Returns (None, 0) if empty.
        """
        # TODO: [Saran perbaikan] Algoritma grouping difflib pada list loop bersarang ini (O(N^2)) bisa menjadi bottleneck jika len(readings) sangat besar. Pertimbangkan algoritma clustering string alternatif yang lebih efisien.
        groups = []
        for r in readings:
            text = r["text"]
            conf = r["conf"]
            matched = False
            for g in groups:
                if difflib.SequenceMatcher(None, g["text"], text).ratio() > 0.8:
                    g["score"] += conf
                    g["texts"].append((text, conf))
                    matched = True
                    break
            if not matched:
                groups.append({"text": text, "score": conf, "texts": [(text, conf)]})
        
        if not groups:
            return None, 0
            
        best_group = max(groups, key=lambda x: x["score"])
        best_exact_text = max(best_group["texts"], key=lambda x: x[1])
        return best_exact_text[0], best_group["score"]

    def process_frame(self, frame, is_video=True):
        """
        Processes a single image frame through the ALPR pipeline:
        Detection -> Tracking -> Rectification -> OCR.

        Args:
            frame (numpy.ndarray): The BGR image frame to process.
            is_video (bool, optional): Whether the frame is part of a video sequence.
                                       Defaults to True.

        Returns:
            tuple: (processed_frame (numpy.ndarray), results (list of dicts)).
        """
        self.clean_expired_plates()
        
        detections = self.detector.detect(frame)
        results = []
        
        rects_with_points = []
        confs = {}
        
        # Memfilter deteksi untuk mengambil objek Nopol (Nomor Polisi) saja
        for det in detections:
            points = det["points"]
            cls_id = det["cls"]
            conf_yolo = det["conf"]
            
            if cls_id == self.nopol_class_id:
                pts_int = np.int32(points)
                x, y, w, h = cv2.boundingRect(pts_int)
                points_f32 = np.array(points, dtype="float32")
                rect = (x, y, w, h)
                rects_with_points.append((rect, points_f32))
                confs[rect] = conf_yolo
        
        tracked_objects = self.tracker.update(rects_with_points)
        current_time = time.time()
        
        for track_id, obj in tracked_objects.items():
            if obj.disappeared > 0:
                continue
                
            points_f32 = obj.points_f32
            rect = obj.bbox
            conf_yolo = confs.get(rect, 0.5)
            
            # Meluruskan (rectify) plat nomor untuk persiapan OCR
            cropped_nopol = rectify_plate(frame, points_f32)
            # Mengenali teks dengan model OCR (CRNN)
            text = self.recognizer.recognize(cropped_nopol).strip()
            
            if len(text) >= 4:
                if track_id not in self.track_buffers:
                    self.track_buffers[track_id] = {
                        "readings": [], 
                        "last_seen": current_time, 
                        "logged": False,
                        "best_img": cropped_nopol,
                        "best_conf": conf_yolo
                    }
                
                self.track_buffers[track_id]["readings"].append({"text": text, "conf": conf_yolo})
                self.track_buffers[track_id]["last_seen"] = current_time
                
                if conf_yolo > self.track_buffers[track_id]["best_conf"]:
                    self.track_buffers[track_id]["best_conf"] = conf_yolo
                    self.track_buffers[track_id]["best_img"] = cropped_nopol
                
                ordered_pts = order_points(points_f32)
                tl, tr, br, bl = ordered_pts
                dx = tr[0] - tl[0]
                dy = tr[1] - tl[1]
                angle_deg = np.degrees(np.arctan2(-dy, dx))
                obb_width = np.linalg.norm(tr - tl)
                obb_height = np.linalg.norm(bl - tl)
                top_center_x = (tl[0] + tr[0]) / 2.0
                top_center_y = (tl[1] + tr[1]) / 2.0
                
                # Menghitung vektor normalisasi (tegak lurus)
                margin = max(10, obb_height * 0.5)
                nx = dy
                ny = -dx
                norm_len = np.linalg.norm([nx, ny])
                if norm_len > 0:
                    nx /= norm_len
                    ny /= norm_len
                else:
                    nx, ny = 0, -1
                    
                text_cx = top_center_x + nx * margin
                text_cy = top_center_y + ny * margin
                font_scale = max(0.4, obb_width / 200.0)
                base_thickness = max(1, int(font_scale * 2))
                
                display_text = f"ID:{track_id} {text}"
                self.draw_tilted_text(
                    img=frame, 
                    text=display_text, 
                    center_x=text_cx, 
                    center_y=text_cy, 
                    angle_deg=angle_deg, 
                    font_scale=font_scale, 
                    thickness=base_thickness, 
                    color=(0, 255, 0),
                    outline_color=(0, 0, 0)
                )

        if is_video:
            for track_id, data in list(self.track_buffers.items()):
                if current_time - data["last_seen"] > 0.5:
                    if not data["logged"] and len(data["readings"]) >= 3:
                        best_text, total_score = self.get_best_text_by_weighted_voting(data["readings"])
                        
                        if best_text and best_text not in self.logged_texts:
                            self.logged_texts[best_text] = current_time
                            self.log_to_file(best_text, data["best_conf"])
                            self.save_plate_image(best_text, data["best_conf"], data["best_img"])
                            
                            results.append({"text": best_text, "conf": data["best_conf"]})
                            data["logged"] = True
                            
                    del self.track_buffers[track_id]
        else:
            # Kondisi gambar diam (still image), langsung log semua hasil bacaan
            for track_id, data in list(self.track_buffers.items()):
                if len(data["readings"]) > 0:
                    best_text, _ = self.get_best_text_by_weighted_voting(data["readings"])
                    if best_text and best_text not in self.logged_texts:
                        self.logged_texts[best_text] = current_time
                        self.log_to_file(best_text, data["best_conf"])
                        self.save_plate_image(best_text, data["best_conf"], data["best_img"])
                        results.append({"text": best_text, "conf": data["best_conf"]})
            self.track_buffers.clear()

        return frame, results
