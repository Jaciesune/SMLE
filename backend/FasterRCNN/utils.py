"""
Moduł narzędziowy dla modelu Faster R-CNN

Ten moduł zawiera funkcje pomocnicze do filtrowania, wizualizacji i przetwarzania
wyników detekcji modelu Faster R-CNN. Implementuje mechanizmy filtrowania ramek
na podstawie pewności detekcji, proporcji i rozmiaru obiektów.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na systemie plików
import cv2               # OpenCV do operacji na obrazach i wizualizacji
import numpy as np       # Do operacji numerycznych i manipulacji tablicami
import sys               # Do operacji systemowych
from config import (     # Import parametrów konfiguracyjnych
    CONFIDENCE_THRESHOLD,  # Minimalny próg pewności dla detekcji
    MAX_BOX_AREA_RATIO,   # Maksymalny stosunek powierzchni ramki do obrazu
    MAX_ASPECT_RATIO      # Maksymalny stosunek szerokości do wysokości (lub odwrotnie)
)

#######################
# Konfiguracja kodowania
#######################
# Wymuszenie kodowania UTF-8 dla stdout
sys.stdout.reconfigure(encoding='utf-8')

#######################
# Funkcje pomocnicze
#######################
def filter_and_draw_boxes(image_np, boxes, scores, image_size):
    """
    Filtruje i rysuje ramki detekcji na obrazie, stosując wiele kryteriów filtrowania.
    
    Funkcja filtruje wyniki detekcji na podstawie:
    1. Progu pewności (określonego przez CONFIDENCE_THRESHOLD)
    2. Maksymalnego stosunku powierzchni ramki do obrazu (określonego przez MAX_BOX_AREA_RATIO)
    3. Maksymalnego współczynnika proporcji (określonego przez MAX_ASPECT_RATIO)
    
    Ramki, które przejdą przez filtry, są rysowane na oryginalnym obrazie.
    
    Args:
        image_np (numpy.ndarray): Obraz w formacie NumPy (H, W, C) w przestrzeni kolorów BGR.
        boxes (list): Lista ramek ograniczających w formacie [x_min, y_min, x_max, y_max].
        scores (list): Lista wartości pewności odpowiadających ramkom.
        image_size (tuple): Wymiary obrazu w formacie (wysokość, szerokość).
        
    Returns:
        tuple: Para (obraz z narysowanymi ramkami, liczba zaakceptowanych ramek).
    """
    h_img, w_img = image_size
    image_area = w_img * h_img
    count = 0

    for box, score in zip(boxes, scores):
        #######################
        # Filtrowanie na podstawie pewności
        #######################
        if score < CONFIDENCE_THRESHOLD:
            continue

        #######################
        # Filtrowanie na podstawie geometrii
        #######################
        # Obliczenie wymiarów i proporcji ramki
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        aspect_ratio = max(width / height, height / width)

        # Sprawdzenie warunków odfiltrowania (zbyt duża powierzchnia lub proporcja)
        if area > MAX_BOX_AREA_RATIO * image_area or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        #######################
        # Rysowanie zaakceptowanych ramek
        #######################
        cv2.rectangle(image_np, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        count += 1

    return image_np, count