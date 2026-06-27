"""
Module providing a robust centroid-based object tracking algorithm.
"""
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

class TrackedObject:
    """
    A representation of a tracked object holding its state and history.

    Attributes:
        id (int): The unique identifier for the object.
        centroid (tuple): The (x, y) coordinates of the object's center.
        bbox (tuple): The bounding box coordinates (x, y, w, h).
        points_f32 (numpy.ndarray): The four corner points of the oriented bounding box.
        disappeared (int): The number of consecutive frames the object has not been seen.
        age (int): The total number of frames the object has been tracked.
    """
    def __init__(self, obj_id, centroid, bbox, points_f32):
        self.id = obj_id
        self.centroid = centroid
        self.bbox = bbox
        self.points_f32 = points_f32
        self.disappeared = 0
        self.age = 1

class CentroidTracker:
    """
    A simple yet effective object tracker based on Euclidean distance between centroids.
    
    Attributes:
        next_object_id (int): The next available unique ID for a newly registered object.
        objects (dict): A dictionary mapping object IDs to TrackedObject instances.
        max_disappeared (int): The maximum consecutive frames an object can be lost before deregistration.
        max_distance (int): The maximum allowed Euclidean distance to associate a detection with an existing track.
    """
    def __init__(self, max_disappeared=5, max_distance=100):
        """
        Initializes the CentroidTracker.

        Args:
            max_disappeared (int, optional): Max frames to keep an object alive without detection. Defaults to 5.
            max_distance (int, optional): Max distance between centroids to consider them the same object. Defaults to 100.
        """
        self.next_object_id = 0
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, points_f32):
        """
        Registers a newly detected object into the tracker.

        Args:
            centroid (tuple): The (x, y) coordinates of the object's center.
            bbox (tuple): The (x, y, w, h) bounding box of the object.
            points_f32 (numpy.ndarray): The specific four points of the object's OBB.
        """
        self.objects[self.next_object_id] = TrackedObject(self.next_object_id, centroid, bbox, points_f32)
        self.next_object_id += 1

    def deregister(self, obj_id):
        """
        Removes an object from the tracking registry.

        Args:
            obj_id (int): The ID of the object to deregister.
        """
        del self.objects[obj_id]

    def update(self, rects_with_points):
        """
        Updates the tracker with new detections for the current frame.

        Args:
            rects_with_points (list): A list of tuples, where each tuple contains:
                                      - rect (tuple): (x, y, w, h)
                                      - pts (numpy.ndarray): (4, 2) array of OBB points.

        Returns:
            dict: The dictionary of currently tracked objects.
        """
        # Jika tidak ada deteksi sama sekali, tingkatkan parameter disappeared semua objek
        if len(rects_with_points) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.zeros((len(rects_with_points), 2), dtype="int")
        input_rects = []
        input_points = []
        
        # Hitung titik tengah (centroid) untuk setiap deteksi yang masuk
        for i, (rect, pts) in enumerate(rects_with_points):
            x, y, w, h = rect
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)
            input_rects.append(rect)
            input_points.append(pts)

        # Jika belum ada objek yang dilacak, registrasi semua deteksi
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_rects[i], input_points[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id].centroid for obj_id in object_ids]

            # Hitung jarak antara centroid objek yang dilacak dengan centroid deteksi input
            D = distance.cdist(np.array(object_centroids), input_centroids)

            # Lakukan penugasan linier optimal menggunakan Hungarian algorithm untuk meminimalisasi cost (jarak)
            row_ind, col_ind = linear_sum_assignment(D)

            used_rows = set()
            used_cols = set()

            for row, col in zip(row_ind, col_ind):
                # Abaikan asosiasi jika jaraknya melebihi threshold
                if D[row, col] > self.max_distance:
                    continue
                
                obj_id = object_ids[row]
                self.objects[obj_id].centroid = input_centroids[col]
                self.objects[obj_id].bbox = input_rects[col]
                self.objects[obj_id].points_f32 = input_points[col]
                self.objects[obj_id].disappeared = 0
                self.objects[obj_id].age += 1
                
                used_rows.add(row)
                used_cols.add(col)

            # Hitung baris (objek terlacak) dan kolom (deteksi baru) yang belum terasosiasi
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Tandai objek terlacak yang tidak terasosiasi (disappeared increment)
            for row in unused_rows:
                obj_id = object_ids[row]
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    self.deregister(obj_id)

            # Register deteksi baru yang tidak terasosiasi sebagai objek baru
            for col in unused_cols:
                self.register(input_centroids[col], input_rects[col], input_points[col])

        return self.objects
