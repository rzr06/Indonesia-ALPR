"""
Utility functions for geometric transformations and image rectification.
"""
import cv2
import numpy as np

def order_points(pts):
    """
    Orders a set of four 2D points in a consistent order:
    top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (numpy.ndarray): An array of shape (4, 2) containing the four points.

    Returns:
        numpy.ndarray: An array of shape (4, 2) containing the ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Titik top-left memiliki jumlah (x+y) terkecil
    rect[0] = pts[np.argmin(s)] 
    # Titik bottom-right memiliki jumlah (x+y) terbesar
    rect[2] = pts[np.argmax(s)]  
    # Titik top-right memiliki selisih (y-x) terkecil
    rect[1] = pts[np.argmin(diff)]  
    # Titik bottom-left memiliki selisih (y-x) terbesar
    rect[3] = pts[np.argmax(diff)]  
    
    return rect


def rectify_plate(image, obb_points):
    """
    Applies a perspective transform to crop and warp an oriented bounding box 
    region into a straightened, upright rectangle.

    Args:
        image (numpy.ndarray): The source image array.
        obb_points (numpy.ndarray): The four points of the oriented bounding box.

    Returns:
        numpy.ndarray: The straightened, cropped image of the license plate.
    """
    # Mengurutkan titik sudut agar proses perspektif lebih konsisten
    pts = order_points(obb_points)

    # Menghitung lebar maksimal antara top-right & top-left atau bottom-right & bottom-left
    w = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))

    # Menghitung tinggi maksimal antara top-left & bottom-left atau top-right & bottom-right
    h = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    ))

    # Definisi koordinat tujuan (bentuk persegi panjang lurus)
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    # Mendapatkan matriks transformasi perspektif
    M = cv2.getPerspectiveTransform(pts, dst)
    # Melakukan warp perspektif ke citra asli
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped
