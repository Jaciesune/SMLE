import torch
import torchvision
import torch_directml

# Przeniesienie modelu na GPU AMD przez DirectML
device = torch_directml.device()

# Wczytanie modelu Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()  # Przełączenie w tryb ewaluacji

print("Model Faster R-CNN załadowany na:", device)
print("Using DirectML:", device)
print(model.roi_heads.detections_per_img)