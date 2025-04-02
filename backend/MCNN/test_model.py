import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from model import MCNN
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys

# Ustawienie urządzenia (GPU lub CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funkcja wczytująca model
def load_model(model_path, device):
    """Wczytuje model MCNN z pliku .pth z końcówką _checkpoint.pth."""
    if not model_path.endswith('_checkpoint.pth'):
        print(f"Błąd: Ścieżka do modelu {model_path} nie kończy się na _checkpoint.pth.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Błąd: Plik modelu {model_path} nie istnieje.")
        sys.exit(1)

    model = MCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

# Przygotowanie transformacji
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Folder wyjściowy
output_folder = "/app/MCNN/data/detectes"
os.makedirs(output_folder, exist_ok=True)

# Funkcja obliczania współczynnika okrągłości
def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

# Uniwersalna funkcja do przetwarzania obrazu
def process_image(image_path, sigma, circularity_range, threshold_factor):
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        density_map = model(img_tensor).cpu().numpy()[0, 0]

    density_map = gaussian_filter(density_map, sigma=sigma)
    threshold = np.mean(density_map) + np.std(density_map) * threshold_factor
    binary_map = (density_map > threshold).astype(np.uint8)
    binary_map_cv = (binary_map * 255).astype(np.uint8)

    contours, _ = cv2.findContours(binary_map_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_cv = np.array(image)
    h_ratio, w_ratio = image_cv.shape[0] / density_map.shape[0], image_cv.shape[1] / density_map.shape[1]

    valid_circles = []
    high_confidence_circles = []

    for contour in contours:
        if len(contour) >= 5:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = calculate_circularity(contour)
            if circularity_range[0] <= circularity <= circularity_range[1]:
                valid_circles.append((contour, (x, y), radius))
                high_confidence_circles.append((contour, (x, y), radius))

    radii = [radius for _, _, radius in high_confidence_circles]
    if radii:
        mean_radius = np.mean(radii)
        min_radius = max(5, mean_radius * 0.55)
        max_radius = mean_radius * 3.0
        valid_circles = [c for c in valid_circles if min_radius <= c[2] <= max_radius]

    marked_contours = 0
    for contour, (x, y), radius in valid_circles:
        x, y, radius = int(x * w_ratio), int(y * h_ratio), int(radius * w_ratio)
        cv2.circle(image_cv, (x, y), radius, (0, 255, 0), 3)
        marked_contours += 1

    return marked_contours, image_cv, density_map

# Wybór lepszej metody
def process_and_choose_best(image_path):
    params_method_1 = (1.5, (0.55, 1.35), 0.5)
    params_method_2 = (2.75, (0.65, 1.35), 0.5)

    marked_contours_1, image_1, density_map_1 = process_image(image_path, *params_method_1)
    marked_contours_2, image_2, density_map_2 = process_image(image_path, *params_method_2)

    if marked_contours_1 >= marked_contours_2:
        return marked_contours_1, image_1, density_map_1
    else:
        return marked_contours_2, image_2, density_map_2

# Funkcja zapisująca wynik
def save_result(image, image_path):
    result_image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_detected.jpg")

    image_cv_resized = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image_cv_resized.shape[0] < 1024 or image_cv_resized.shape[1] < 1024:
        image_cv_resized = cv2.resize(image_cv_resized, (1024, 1024))
    cv2.imwrite(result_image_path, image_cv_resized)

    return result_image_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie: python test_model.py <ścieżka_do_obrazu> <ścieżka_do_modelu>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    # Wczytanie modelu
    model = load_model(model_path, device)

    # Wybór najlepszej metody
    detections_count, best_image, density_map = process_and_choose_best(image_path)

    # Zapis wyniku
    result_path = save_result(best_image, image_path)

    # Wypisanie liczby detekcji w formacie łatwym do sparsowania
    print(f"Detections: {detections_count}")
    print(f"Wynik zapisany pod: {result_path}")
