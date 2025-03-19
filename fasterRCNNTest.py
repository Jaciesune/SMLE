import torch
import torchvision.models.detection
import os
import cv2
import numpy as np
from dataLoader import get_data_loaders

# Pobranie modelu Faster R-CNN
def load_model(model_path, num_classes=2, device="cpu"):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Funkcja testowania modelu
def test_model(model, dataloader, device, model_name):
    save_path = f"test/{model_name}"
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for idx, (image, _) in enumerate(dataloader):  
            image = image.to(device)  # Przenosimy obraz na CPU/GPU
            image_list = [image.squeeze(0)]  # Zamieniamy tensor na listę pojedynczego obrazu

            output = model(image_list)[0]  # Model oczekuje listy, a nie pojedynczego tensora

            image_np = (image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Rysowanie predykcji modelu
            for box in output["boxes"]:
                x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
                cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            filename = f"{save_path}/img_{idx}.png"
            cv2.imwrite(filename, image_np)
            print(f"Zapisano wynik testu: {filename}")

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

# Wybór modelu przez użytkownika
def choose_model(models):
    while True:
        choice = input("\nWybierz numer modelu do testowania: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Błędny wybór, spróbuj ponownie.")

# Uruchomienie testu
if __name__ == "__main__":
    print("\nŁadowanie zbioru testowego...")
    _, _, test_loader = get_data_loaders(batch_size=1, num_workers=4)  # Ustawiamy batch_size=1 dla pojedynczego obrazu
    print(f"DataLoader załadowany: Testowe obrazy: {len(test_loader.dataset)}")

    available_models = list_saved_models()
    selected_model = choose_model(available_models)

    device = torch.device("cpu")
    model = load_model(f"saved_models/{selected_model}", device=device)

    print(f"\nZaładowano model: {selected_model}")
    print("Rozpoczynam testowanie modelu...")

    test_model(model, test_loader, device, selected_model.split(".pth")[0])
    print("\nTestowanie zakończone!")
