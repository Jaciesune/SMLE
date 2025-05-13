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
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_CONTOUR_AREA = 15

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

output_folder = "/app/backend/MCNN/data/detectes"
maps_folder = "/app/backend/MCNN/maps"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(maps_folder, exist_ok=True)

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

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def threshold_by_kmeans(density_map, k=2):
    flat = density_map.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(flat)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = (centers[0] + centers[1]) / 2
    binary_map = (density_map > threshold).astype(np.uint8)
    return binary_map

def process_image(image_path, sigma, threshold_factor=None, resize_shape=(2048, 2048)):
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
    binary_map = threshold_by_kmeans(density_map, k=2)
    binary_map_cv = (binary_map * 255).astype(np.uint8)

    contours, _ = cv2.findContours(binary_map_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_cv = np.array(image)
    h_ratio, w_ratio = image_cv.shape[0] / density_map.shape[0], image_cv.shape[1] / density_map.shape[1]

    valid_contours = []
    for contour in contours:
        if len(contour) >= 5 and cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
            valid_contours.append(contour)

    marked_contours = 0
    marked_centroids = []

    # Ustalanie dynamicznych wartości dla parametrów
    detections_count = len(valid_contours)

    if detections_count < 50:
        MIN_ELLIPSE_AREA = 50
        MAX_ELLIPSE_AREA = 25000
        MIN_DISTANCE_BETWEEN_CENTROIDS = 75
    elif detections_count < 75:
        MIN_ELLIPSE_AREA = 2500
        MAX_ELLIPSE_AREA = 10000
        MIN_DISTANCE_BETWEEN_CENTROIDS = 200
    elif detections_count < 150:
        MIN_ELLIPSE_AREA = 250
        MAX_ELLIPSE_AREA = 10000
        MIN_DISTANCE_BETWEEN_CENTROIDS = 50
    elif detections_count < 500:
        MIN_ELLIPSE_AREA = 100
        MAX_ELLIPSE_AREA = 2500
        MIN_DISTANCE_BETWEEN_CENTROIDS = 25
    else:
        MIN_ELLIPSE_AREA = 5
        MAX_ELLIPSE_AREA = 1000
        MIN_DISTANCE_BETWEEN_CENTROIDS = 5

    ellipses_info = []

    for contour in valid_contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            area = np.pi * (axes[0] / 2) * (axes[1] / 2)

            major_axis = max(axes)
            minor_axis = min(axes)

            if minor_axis == 0:
                continue

            aspect_ratio = major_axis / minor_axis

            if aspect_ratio > 3.5 or area < MIN_ELLIPSE_AREA or area > MAX_ELLIPSE_AREA:
                continue

            ellipses_info.append({
                "contour": contour,
                "ellipse": ellipse,
                "area": area,
                "center": center
            })

    ellipses_info.sort(key=lambda e: e["center"][0])

    filtered_ellipses = []
    for i, current in enumerate(ellipses_info):
        area = current["area"]

        neighbor_areas = []
        if i > 0:
            neighbor_areas.append(ellipses_info[i - 1]["area"])
        if i < len(ellipses_info) - 1:
            neighbor_areas.append(ellipses_info[i + 1]["area"])

        if neighbor_areas:
            avg_neighbors = sum(neighbor_areas) / len(neighbor_areas)
            if area < 0.5 * avg_neighbors:
                continue

        filtered_ellipses.append(current)

    for item in filtered_ellipses:
        ellipse = item["ellipse"]
        (center, axes, angle) = ellipse

        x, y = int(center[0] * w_ratio), int(center[1] * h_ratio)
        axes_scaled = (int(axes[0] * w_ratio / 2), int(axes[1] * h_ratio / 2))

        centroid = calculate_centroid(item["contour"])
        if centroid:
            cx, cy = int(centroid[0] * w_ratio), int(centroid[1] * h_ratio)

            too_close = any(np.linalg.norm(np.array((cx, cy)) - np.array(existing)) < MIN_DISTANCE_BETWEEN_CENTROIDS
                            for existing in marked_centroids)
            if too_close:
                continue

            marked_centroids.append((cx, cy))

            cv2.ellipse(image_cv, (x, y), axes_scaled, angle, 0, 360, (0, 255, 0), 2)
            cv2.circle(image_cv, (cx, cy), 4, (255, 0, 0), -1)

            marked_contours += 1

    return marked_contours, image_cv, density_map



def process_and_choose_best(image_path, resize_shape=(2048, 2048)):
    params_method_1 = (1.5, None)
    params_method_2 = (2.75, None)

    marked_1, img_1, map_1 = process_image(image_path, *params_method_1, resize_shape=resize_shape)
    marked_2, img_2, map_2 = process_image(image_path, *params_method_2, resize_shape=resize_shape)

    if marked_1 >= marked_2:
        return marked_1, img_1, map_1
    else:
        return marked_2, img_2, map_2

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

    model = load_model(model_path, device)

    detections_count, best_image, density_map = process_and_choose_best(image_path, resize_shape=(1024, 1024))

    if detections_count < 100:
        print("Liczba wykrytych obiektów < 100. Przetwarzanie ponownie w 512x512...")
        detections_count, best_image, density_map = process_and_choose_best(image_path, resize_shape=(512, 512))
    elif detections_count > 500:
        print("Liczba wykrytych obiektów > 500. Przetwarzanie ponownie w 2048x2048...")
        detections_count, best_image, density_map = process_and_choose_best(image_path, resize_shape=(2048, 2048))

    result_path = save_result(best_image, image_path)
    save_density_map(density_map, image_path)

    print(f"Detections: {detections_count}")
    print(f"Wynik zapisany pod: {result_path}")
