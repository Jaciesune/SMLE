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
model = MCNN().to(device)
model.load_state_dict(torch.load("object_counting_model.pth", map_location=device))
model.eval()

# Przygotowanie transformacji
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Folder wejściowy i wyjściowy
input_folder = "backend/MCNN/dataset/test/images"
output_folder = "backend/MCNN/dataset/results"

# Funkcja czyszcząca folder results
def clear_results_folder():
    # Sprawdzamy, czy folder istnieje
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Nie udało się usunąć pliku {file_path}: {e}")

# Funkcja obliczania współczynnika okrągłości
def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

# Funkcja przetwarzania obrazów (Metoda 1)
def process_image_method_1(image_path):
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        density_map = model(img_tensor).cpu().numpy()[0, 0]
    
    density_map = gaussian_filter(density_map, sigma=1.5)
    
    threshold = np.mean(density_map) + np.std(density_map) * 0.5  
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
            if 0.55 <= circularity <= 1.35:
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
        cv2.circle(image_cv, (x, y), radius, (255, 0, 0), 3)
        marked_contours += 1
    
    # Zapisz mapę gęstości i wynik metody 1 w folderze results
    density_map_path = os.path.join(output_folder, f"density_map_method_1_{os.path.basename(image_path)}")
    plt.imsave(density_map_path, density_map, cmap='jet')
    
    result_image_path = os.path.join(output_folder, f"result_method_1_{os.path.basename(image_path)}")
    image_cv_resized = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    if image_cv_resized.shape[0] < 1024 or image_cv_resized.shape[1] < 1024:
        image_cv_resized = cv2.resize(image_cv_resized, (1024, 1024))
    cv2.imwrite(result_image_path, image_cv_resized)
    
    return marked_contours, density_map_path, result_image_path

# Funkcja przetwarzania obrazów (Metoda 2)
def process_image_method_2(image_path):
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        density_map = model(img_tensor).cpu().numpy()[0, 0]
    
    density_map = gaussian_filter(density_map, sigma=2.75)  # Większe sigma dla lepszego rozmycia
    
    threshold = np.max(density_map) * 0.5  
    binary_map = (density_map > threshold).astype(np.uint8)
    binary_map_cv = (binary_map * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_map_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_cv = np.array(image)
    h_ratio = image_cv.shape[0] / density_map.shape[0]
    w_ratio = image_cv.shape[1] / density_map.shape[1]
    
    valid_circles = []
    high_confidence_circles = []

    for contour in contours:
        if len(contour) >= 5:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = calculate_circularity(contour)
            if 0.65 <= circularity <= 1.35:
                valid_circles.append((contour, (x, y), radius))
                high_confidence_circles.append((contour, (x, y), radius))
    
    radii = [radius for _, _, radius in high_confidence_circles]
    
    if len(radii) > 0:
        mean_radius = np.mean(radii)
        min_radius = mean_radius * 0.5
        max_radius = mean_radius * 3.0
        valid_circles = [circle for circle in valid_circles if min_radius <= circle[2] <= max_radius]
    
    marked_contours = 0
    for contour, (x, y), radius in valid_circles:
        x, y, radius = int(x * w_ratio), int(y * h_ratio), int(radius * w_ratio)
        cv2.circle(image_cv, (x, y), radius, (255, 0, 0), 3)
        marked_contours += 1
    
    # Zapisz mapę gęstości i wynik metody 2 w folderze results
    density_map_path = os.path.join(output_folder, f"density_map_method_2_{os.path.basename(image_path)}")
    plt.imsave(density_map_path, density_map, cmap='jet')
    
    result_image_path = os.path.join(output_folder, f"result_method_2_{os.path.basename(image_path)}")
    image_cv_resized = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    if image_cv_resized.shape[0] < 1024 or image_cv_resized.shape[1] < 1024:
        image_cv_resized = cv2.resize(image_cv_resized, (1024, 1024))
    cv2.imwrite(result_image_path, image_cv_resized)
    
    return marked_contours, density_map_path, result_image_path

# Funkcja przetwarzania obrazów
def process_image(image_path):
    # Przetwarzamy oba obrazy (metoda 1 i metoda 2)
    marked_contours_method_1, density_map_method_1, result_method_1 = process_image_method_1(image_path)
    marked_contours_method_2, density_map_method_2, result_method_2 = process_image_method_2(image_path)

    # Porównanie wyników obu metod i zapisanie tylko lepszej metody
    if marked_contours_method_1 >= marked_contours_method_2:
        print(f"Przetworzono {image_path}: Wybrano metodę 1 z {marked_contours_method_1} obiektami.")
        # Zapisz tylko wyniki z metody 1
        os.rename(density_map_method_1, os.path.join(output_folder, f"density_map_{os.path.basename(image_path)}"))
        os.rename(result_method_1, os.path.join(output_folder, f"result_{os.path.basename(image_path)}"))
    else:
        print(f"Przetworzono {image_path}: Wybrano metodę 2 z {marked_contours_method_2} obiektami.")
        # Zapisz tylko wyniki z metody 2
        os.rename(density_map_method_2, os.path.join(output_folder, f"density_map_{os.path.basename(image_path)}"))
        os.rename(result_method_2, os.path.join(output_folder, f"result_{os.path.basename(image_path)}"))

# Funkcja usuwająca pliki zaczynające się od "result_method" lub "density_map_method"
def remove_result_method_files():
    for filename in os.listdir(output_folder):
        # Sprawdzamy, czy nazwa pliku zaczyna się od "result_method" lub "density_map_method"
        if filename.startswith("result_method") or filename.startswith("density_map_method"):
            file_path = os.path.join(output_folder, filename)
            try:
                os.remove(file_path)
                #print(f"Usunięto {filename}")
            except Exception as e:
                print(f"Nie udało się usunąć pliku {file_path}: {e}")


# Przetwarzanie wszystkich obrazów w folderze
clear_results_folder()  # Czyścimy folder przed rozpoczęciem
for image_name in os.listdir(input_folder):
    if image_name.endswith(".jpg"):
        image_path = os.path.join(input_folder, image_name)
        process_image(image_path)

# Usuwanie plików zaczynających się od "result_method"
remove_result_method_files()

print("✅ Wszystkie obrazy zostały przetworzone!")
