import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from model import MCNN
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MCNN().to(device)
model.load_state_dict(torch.load("object_counting_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  
    transforms.ToTensor()
])

input_folder = "backend/MCNN/dataset/test/images"
output_folder = "backend/MCNN/dataset/results"
os.makedirs(output_folder, exist_ok=True)

def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

def process_image(image_path):
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
            if 0.5 <= circularity <= 1.35:
                valid_circles.append((contour, (x, y), radius))
                high_confidence_circles.append((contour, (x, y), radius))
    
    radii = [radius for _, _, radius in high_confidence_circles]
    if radii:
        mean_radius = np.mean(radii)
        min_radius = max(5, mean_radius * 0.85)
        max_radius = mean_radius * 3.0
        valid_circles = [c for c in valid_circles if min_radius <= c[2] <= max_radius]
    
    marked_contours = 0
    for contour, (x, y), radius in valid_circles:
        x, y, radius = int(x * w_ratio), int(y * h_ratio), int(radius * w_ratio)
        cv2.circle(image_cv, (x, y), radius, (255, 0, 0), 3)
        marked_contours += 1
    
    density_map_path = os.path.join(output_folder, f"density_map_{os.path.basename(image_path)}")
    plt.imsave(density_map_path, density_map, cmap='jet')
    
    result_image_path = os.path.join(output_folder, f"result_{os.path.basename(image_path)}")
    image_cv_resized = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    if image_cv_resized.shape[0] < 1024 or image_cv_resized.shape[1] < 1024:
        image_cv_resized = cv2.resize(image_cv_resized, (1024, 1024))
    cv2.imwrite(result_image_path, image_cv_resized)
    
    print(f"Przetworzono {image_path}, wykryto i oznaczono {marked_contours} rur.")

for image_name in os.listdir(input_folder):
    if image_name.endswith(".jpg"):
        process_image(os.path.join(input_folder, image_name))

print("✅ Wszystkie obrazy zostały przetworzone!")
