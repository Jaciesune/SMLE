"""
Moduł preprocesingu obrazów dla modeli detekcji obiektów

Ten skrypt implementuje zaawansowane techniki przetwarzania obrazów w celu
uwydatnienia kluczowych cech, usunięcia szumów i poprawy kontrastu. Szczególnie
przydatny w przygotowaniu obrazów do detekcji małych obiektów lub obiektów
o niskim kontraście.
"""

#######################
# Importy bibliotek
#######################
import cv2               # OpenCV do operacji na obrazach
import numpy as np       # Biblioteka do operacji numerycznych
import os                # Do operacji na systemie plików
import sys               # Do obsługi argumentów wiersza poleceń

#######################
# Funkcje przetwarzania obrazów
#######################
def resize_with_aspect_ratio(image, target_size=2048):
    """
    Zmienia rozmiar obrazu zachowując proporcje.
    
    Funkcja inteligentnie wybiera metodę interpolacji w zależności od tego,
    czy obraz jest powiększany czy pomniejszany.
    
    Args:
        image (numpy.ndarray): Obraz wejściowy w formacie OpenCV.
        target_size (int): Docelowy rozmiar dłuższego boku.
        
    Returns:
        numpy.ndarray: Przeskalowany obraz.
    """
    h, w = image.shape[:2]
    if h < target_size or w < target_size:
        # Powiększanie obrazu - używamy LANCZOS4 dla lepszej jakości
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Pomniejszanie obrazu - używamy INTER_AREA dla lepszej jakości
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_image(image_path, output_path=None, output_size=(2048, 2048)):
    """
    Główna funkcja preprocesingu wykonująca wieloetapowe przetwarzanie obrazu.
    
    Proces obejmuje:
    1. Zmianę rozmiaru z zachowaniem proporcji
    2. Redukcję szumu za pomocą filtrów
    3. Poprawę kontrastu przy użyciu CLAHE
    4. Normalizację wartości pikseli
    5. Usunięcie cieni
    6. Detekcję krawędzi i uwydatnienie szczegółów
    7. Operacje morfologiczne
    8. Usunięcie małych obiektów
    
    Args:
        image_path (str): Ścieżka do obrazu wejściowego.
        output_path (str, optional): Ścieżka wyjściowa (jeśli inna niż wejściowa).
        output_size (tuple): Docelowy rozmiar obrazu (szerokość, wysokość).
        
    Returns:
        numpy.ndarray: Przetworzony obraz lub None w przypadku błędu.
    """
    # Wczytanie obrazu
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: nie można wczytać {image_path}")
        return None

    # Zmiana rozmiaru
    img_resized = resize_with_aspect_ratio(img, target_size=output_size[0])
    
    #######################
    # Redukcja szumów
    #######################
    # Zastosowanie filtra Gaussa i medianowego
    img_smoothed = cv2.GaussianBlur(img_resized, (3, 3), 0)
    img_smoothed = cv2.medianBlur(img_smoothed, 3)
    
    #######################
    # Poprawa kontrastu
    #######################
    # Konwersja do skali szarości
    gray = cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2GRAY)
    
    # Zastosowanie CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(64, 64))
    clahe_img = clahe.apply(gray)
    
    #######################
    # Normalizacja i usuwanie cieni
    #######################
    # Normalizacja wartości pikseli
    norm_img = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Identyfikacja i usunięcie ciemnych cieni
    _, dark_shadows = cv2.threshold(norm_img, 50, 255, cv2.THRESH_BINARY_INV)
    norm_img = cv2.subtract(norm_img, dark_shadows)
    
    #######################
    # Binaryzacja i detekcja krawędzi
    #######################
    # Dodatkowe rozmycie dla stabilizacji progowania
    blurred = cv2.GaussianBlur(norm_img, (3, 3), 0)
    
    # Dwie maski binarne dla różnych poziomów jasności
    _, binary_dark = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY)
    _, binary_light = cv2.threshold(blurred, 125, 100, cv2.THRESH_BINARY_INV)
    
    # Połączenie masek z różnymi wagami
    combined = cv2.addWeighted(binary_dark, 0.75, binary_light, 0.25, 0)
    
    # Detekcja krawędzi i połączenie z obrazem binarnym
    edges = cv2.Canny(combined, 125, 175)
    final_image = cv2.addWeighted(combined, 0.95, edges, 0.25, 0)
    
    #######################
    # Operacje morfologiczne
    #######################
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    #######################
    # Usuwanie małych obiektów
    #######################
    # Wykrywanie konturów
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Usuwanie konturów o powierzchni mniejszej niż min_area
    min_area = 500
    for c in contours:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(opened, [c], -1, 0, -1)

    return opened

#######################
# Punkt wejścia programu
#######################
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else image_path
        # Utwórz katalog wyjściowy, jeśli nie istnieje
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed = preprocess_image(image_path, output_path)
        if processed is not None:
            cv2.imwrite(output_path, processed)
            print(f"Zapisano preprocesowany obraz: {output_path}")
        else:
            print("Błąd: Nie udało się przetworzyć obrazu.")
            sys.exit(1)
    else:
        print("Użycie: python preprocessing.py /ścieżka/do/obrazu.jpg [/ścieżka/wyjściowa]")
        sys.exit(1)