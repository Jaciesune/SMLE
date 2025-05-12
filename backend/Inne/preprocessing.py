import cv2
import numpy as np
import os
import sys

def resize_with_aspect_ratio(image, target_size=2048):
    h, w = image.shape[:2]
    if h < target_size or w < target_size:
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_image(image_path, output_path=None, output_size=(2048, 2048)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: nie można wczytać {image_path}")
        return None

    img_resized = resize_with_aspect_ratio(img, target_size=output_size[0])
    img_smoothed = cv2.GaussianBlur(img_resized, (3, 3), 0)
    img_smoothed = cv2.medianBlur(img_smoothed, 3)
    gray = cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(64, 64))
    clahe_img = clahe.apply(gray)
    norm_img = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    _, dark_shadows = cv2.threshold(norm_img, 50, 255, cv2.THRESH_BINARY_INV)
    norm_img = cv2.subtract(norm_img, dark_shadows)
    blurred = cv2.GaussianBlur(norm_img, (3, 3), 0)
    _, binary_dark = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY)
    _, binary_light = cv2.threshold(blurred, 125, 100, cv2.THRESH_BINARY_INV)
    combined = cv2.addWeighted(binary_dark, 0.75, binary_light, 0.25, 0)
    edges = cv2.Canny(combined, 125, 175)
    final_image = cv2.addWeighted(combined, 0.95, edges, 0.25, 0)
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    for c in contours:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(opened, [c], -1, 0, -1)

    return opened

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