import torch
import torchvision
import os

# Ścieżka docelowa
output_dir = "../pretrained_weights"
os.makedirs(output_dir, exist_ok=True)

# Pobranie modelu z domyślnymi wagami
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

# Ścieżka do zapisu
weights_path = os.path.join(output_dir, "maskrcnn_resnet50_fpn_v2.pth")

# Zapis wag do pliku
torch.save(model.state_dict(), weights_path)

print(f"Wagi zostały zapisane w: {weights_path}")
