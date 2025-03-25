import time
import torch
import torchvision
import torch_directml
from PIL import Image
from torchvision.transforms import functional as F

# Przeniesienie modelu na DirectML
device = torch_directml.device(0)

# Wczytanie modelu
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()

# Testowe zdjęcie
image = Image.open("./dataset/test/images/1.jpg")  # Podmień na dowolny obraz testowy
image = F.to_tensor(image).unsqueeze(0).to(device)  # Konwersja do tensora i przeniesienie na DirectML

# Test wydajności
start_time = time.time()
output = model(image)
end_time = time.time()

print(f"Przetwarzanie zajęło: {end_time - start_time:.4f} s")
print(f"Liczba wykrytych obiektów: {len(output[0]['boxes'])}")
