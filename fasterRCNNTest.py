import torch
import torchvision.models.detection
import os
import cv2
import numpy as np
from dataLoader import get_data_loaders

# Pobranie listy zapisanych modeli
def list_saved_models():
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    
    if not models:
        print("Brak zapisanych modeli! Najpierw przeprowadź trening.")
        exit(1)
    
    print("\nDostępne modele:")
    for idx, model_name in enumerate(models, 1):
        print(f"{idx}. {model_name}")
    
    return models

# Wybór modelu
def choose_model(models):
    while True:
        choice = input("Wybierz numer modelu do testowania: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        else:
            print("Błędny wybór, spróbuj ponownie.")

# Wczytanie modelu Faster R-CNN
def load_model(model_path, num_classes=2, device="cpu"):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Załadowano model: {model_path}")
    return model

# Funkcja testowania modelu
def test_model(model, dataloader, device, model_name):
    model.eval()
    os.makedirs("test", exist_ok=True)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, (image, output) in enumerate(zip(images, outputs)):
                image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Poprawiony format obrazu

                for box in output["boxes"]:
                    x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                filename = f"test/{model_name}_result_{idx}_{i}.png"
                cv2.imwrite(filename, image_np)
                print(f"Zapisano wynik testu: {filename}")

if __name__ == "__main__":
    print("Ładowanie zbioru testowego...")
    _, _, test_loader = get_data_loaders(batch_size=2, num_workers=4)
    print("Dane załadowane!")

    available_models = list_saved_models()
    selected_model = choose_model(available_models)

    device = torch.device("cpu")
    model = load_model(f"saved_models/{selected_model}", device=device)

    print("\nRozpoczynam testowanie modelu...\n")
    test_model(model, test_loader, device, selected_model.split(".pth")[0])
    print("\nTestowanie zakończone!")