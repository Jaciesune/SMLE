import os
import torch
from glob import glob
from PIL import Image
import numpy as np
import torchvision.transforms as T
from model import get_model
from config import CONFIDENCE_THRESHOLD
from utils import filter_and_draw_boxes
import cv2

def list_models():
    model_paths = glob("saved_models/*/model_*.pth")
    model_paths = sorted(model_paths)
    if not model_paths:
        print("Nie znaleziono zapisanych modeli w saved_models/*/model_*.pth")
        exit(1)

    print("Dostępne modele Faster R-CNN:")
    for i, path in enumerate(model_paths):
        print(f"  [{i}] {path}")
    return model_paths

def main():
    # Wybór modelu
    model_paths = list_models()
    while True:
        try:
            idx = int(input("Wybierz numer modelu do załadowania: "))
            model_path = model_paths[idx]
            break
        except (ValueError, IndexError):
            print("Niepoprawny wybór. Spróbuj ponownie.")

    # Wybór obrazu
    image_paths = glob("dataset/test/*.*")
    if not image_paths:
        print("Nie znaleziono obrazów w dataset/test/")
        exit(1)

    print("\nDostępne obrazy testowe:")
    for i, path in enumerate(image_paths):
        print(f"  [{i}] {path}")

    while True:
        try:
            img_idx = int(input("Wybierz numer obrazu do testu: "))
            image_path = image_paths[img_idx]
            break
        except (ValueError, IndexError):
            print("Niepoprawny wybór. Spróbuj ponownie.")

    # Wczytanie modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nŁadowanie modelu z {model_path} ({device})")

    model = get_model(num_classes=2, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Wczytanie obrazu
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    # Filtrujemy wykrycia
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    image_np = np.array(image)
    image_np, count = filter_and_draw_boxes(image_np, boxes, scores, image_np.shape[:2])

    # Zapis wyniku
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = "test/faster_predict"
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(f"{save_dir}/{base_name}.jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    with open(f"{save_dir}/{base_name}.txt", "w") as f:
        f.write(f"{count}\n")

    print(f"\nNa obrazie '{base_name}' wykryto {count} rur.")
    print(f"Wyniki zapisane w: {save_dir}/{base_name}.jpg + .txt")

if __name__ == "__main__":
    main()
