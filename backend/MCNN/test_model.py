import sys
print("Interpreter:", sys.executable)

import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from model import MCNN
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Ustawienie urządzenia (GPU lub CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funkcja wczytująca model
def load_model(model_path, device):
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

# Foldery wyjściowe
output_folder = "/app/backend/MCNN/data/detectes"
maps_folder = "/app/backend/MCNN/maps"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(maps_folder, exist_ok=True)

# Funkcja zapisywania mapy gęstości
def save_density_map(density_map, image_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.title('Density Map')
    plt.axis('off')
    map_filename = os.path.splitext(os.path.basename(image_path))[0] + '_density_map.png'
    map_save_path = os.path.join(maps_folder, map_filename)
    plt.savefig(map_save_path, bbox_inches='tight')
    plt.close()
    print(f"Mapa gęstości zapisana pod: {map_save_path}")

# Obliczanie współczynnika okrągłości
def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

# Przetwarzanie obrazu
def process_image(image_path, sigma, circularity_range, threshold_factor, resize_shape=(1024, 1024)):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    resize_transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor()
    ])
    img_tensor = resize_transform(image).unsqueeze(0).to(device)

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

# Wybór najlepszej metody
def process_and_choose_best(image_path, resize_shape=(1024, 1024)):
    params_method_1 = (1.5, (0.55, 1.35), 0.5)
    params_method_2 = (2.75, (0.65, 1.35), 0.5)

    marked_1, img_1, map_1 = process_image(image_path, *params_method_1, resize_shape=resize_shape)
    marked_2, img_2, map_2 = process_image(image_path, *params_method_2, resize_shape=resize_shape)

    if marked_1 >= marked_2:
        return marked_1, img_1, map_1
    else:
        return marked_2, img_2, map_2

# Zapis wyniku
def save_result(image, image_path):
    result_image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_detected.jpg")
    image_cv_resized = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image_cv_resized.shape[0] < 1024 or image_cv_resized.shape[1] < 1024:
        image_cv_resized = cv2.resize(image_cv_resized, (1024, 1024))
    cv2.imwrite(result_image_path, image_cv_resized)
    return result_image_path

# Główne wywołanie
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie: python test_model.py <ścieżka_do_obrazu> <ścieżka_do_modelu>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    # Wczytanie modelu
    model = load_model(model_path, device)

    # Przetwarzanie w domyślnej rozdzielczości
    detections_count, best_image, density_map = process_and_choose_best(image_path, resize_shape=(1024, 1024))

    # Dodatkowe przetwarzanie w zależności od liczby wykryć
    if detections_count < 100:
        print("Liczba wykrytych obiektów < 100. Przetwarzanie ponownie w 512x512...")
        detections_count, best_image, density_map = process_and_choose_best(image_path, resize_shape=(512, 512))

    elif detections_count > 500:
        print("Liczba wykrytych obiektów > 500. Przetwarzanie ponownie w 2048x2048...")
        detections_count, best_image, density_map = process_and_choose_best(image_path, resize_shape=(2048, 2048))

    # Zapis wyników
    result_path = save_result(best_image, image_path)
    save_density_map(density_map, image_path)

    print(f"Detections: {detections_count}")
    print(f"Wynik zapisany pod: {result_path}")
