import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from model import MCNN
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Sprawdź, czy GPU jest dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Załaduj model i przenieś go na GPU (lub CPU, jeśli GPU nie jest dostępne)
model = MCNN().to(device)
model.load_state_dict(torch.load("object_counting_model.pth", map_location=device, weights_only=True))
model.eval()

# Przygotowanie transformacji
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  
    transforms.ToTensor()
])

# Foldery wejściowe i wyjściowe
input_folder = "dataset/test/images"
output_folder = "dataset/results"
os.makedirs(output_folder, exist_ok=True)  # Tworzymy folder, jeśli nie istnieje

# Funkcja obliczania współczynnika okrągłości
def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

# Funkcja przetwarzania obrazów
def process_image(image_path):
    # Załaduj obraz testowy
    image = Image.open(image_path)

    # Zastosuj transformację
    img_tensor = transform(image).unsqueeze(0).to(device)  # Przenosimy tensor na GPU

    # Przetwarzanie przez model
    with torch.no_grad():
        density_map = model(img_tensor).cpu().numpy()[0, 0]  # Zwracamy wynik na CPU, jeśli był na GPU

    # **Krok 1: Wygładzenie mapy gęstości z mniejszym sigma**
    density_map = gaussian_filter(density_map, sigma=2.75)  # Większe sigma dla lepszego rozmycia

    # **Krok 2: Binaryzacja mapy gęstości z bardziej dynamicznym progiem**
    threshold = np.max(density_map) * 0.5  # Ustawienie progu jako połowa maksymalnej wartości w mapie
    binary_map = (density_map > threshold).astype(np.uint8)  # Przekształcenie na obraz binarny (0/1)

    # **Krok 3: Znalezienie konturów**
    binary_map_cv = (binary_map * 255).astype(np.uint8)  # Konwersja na format OpenCV
    contours, _ = cv2.findContours(binary_map_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **Krok 4: Wybór elementów w odpowiednim zakresie rozmiaru**
    image_cv = np.array(image)
    h_ratio = image_cv.shape[0] / density_map.shape[0]
    w_ratio = image_cv.shape[1] / density_map.shape[1]

    # Określenie "referencyjnego rozmiaru" na podstawie obiektów z wysoką pewnością (powyżej 95%)
    marked_contours = 0  # Liczba oznaczonych elementów
    valid_circles = []  # Lista okręgów o wysokiej pewności
    high_confidence_circles = []  # Lista obiektów z wysoką pewnością

    for contour in contours:
        if len(contour) >= 5:  # Aby uniknąć przypadkowych punktów
            # Obliczanie okręgu minimalnego wokół konturu
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # Obliczanie współczynnika okrągłości
            circularity = calculate_circularity(contour)

            # Jeśli kontur jest wystarczająco okrągły, przechowuj go
            if 0.65 <= circularity <= 1.35:  # Zaokrąglone kształty (można dostosować)
                valid_circles.append((contour, (x, y), radius))
                
                # Dodaj do listy obiektów z wysoką pewnością
                high_confidence_circles.append((contour, (x, y), radius))

    # Oblicz średni promień z obiektów o wysokiej pewności (powyżej 95%)
    radii = [radius for _, _, radius in high_confidence_circles]
    
    # Sprawdź, czy mamy obiekty o wysokiej pewności
    if len(radii) > 0:
        mean_radius = np.mean(radii)
        # Zastosowanie zakresu wahań (0.7 do 1.7 średniego promienia)
        min_radius = mean_radius * 0.65
        max_radius = mean_radius * 2.25

        # Wybieranie tylko obiektów mieszczących się w tym zakresie
        valid_circles = [circle for circle in valid_circles if min_radius <= circle[2] <= max_radius]

    # Teraz zliczamy tylko te elementy, które są okrągłe i mają rozmiar w zakresie
    for contour, (x, y), radius in valid_circles:
        x, y, radius = int(x * w_ratio), int(y * h_ratio), int(radius * w_ratio)
        # Rysowanie okręgu wokół rury
        cv2.circle(image_cv, (x, y), radius, (255, 0, 0), 3)
        marked_contours += 1  # Zwiększ liczbę oznaczonych elementów

    # Zapisz mapę gęstości do wyników
    density_map_path = os.path.join(output_folder, f"density_map_{os.path.basename(image_path)}")
    plt.imsave(density_map_path, density_map, cmap='jet')

    # Zapisz obraz z okręgami
    result_image_path = os.path.join(output_folder, f"result_{os.path.basename(image_path)}")
    
    # **Zmiana rozmiaru obrazu, jeśli jest mniejszy niż 1024x1024**
    image_cv_resized = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    if image_cv_resized.shape[0] < 1024 or image_cv_resized.shape[1] < 1024:
        image_cv_resized = cv2.resize(image_cv_resized, (1024, 1024))  # Zmieniamy rozmiar do 1024x1024
    
    # Zapisz ostateczny obraz z okręgami
    cv2.imwrite(result_image_path, image_cv_resized)

    # Wypisanie liczby wykrytych i oznaczonych elementów
    print(f"Przetworzono {image_path}, wykryto i oznaczono {marked_contours} rur.")
    
# 🔹 Przetwarzanie wszystkich obrazów w folderze
for image_name in os.listdir(input_folder):
    if not image_name.endswith(".jpg"):
        continue  # Pomijamy pliki inne niż JPG

    image_path = os.path.join(input_folder, image_name)
    process_image(image_path)

print("✅ Wszystkie obrazy zostały przetworzone!")
