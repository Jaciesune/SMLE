import torch
import torchvision
import torch_directml

# Przeniesienie modelu na GPU AMD przez DirectML
device = torch_directml.device(0)

print("Dostępne urządzenia DirectML:", torch_directml.device_count())
print("Aktualne urządzenie:", device)


# Wczytanie modelu Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()  # Przełączenie w tryb ewaluacji

print("Czy model na GPU?", next(model.parameters()).device)

print("Model Faster R-CNN załadowany na:", device)
print("Using DirectML:", device)
print(torch.cuda.is_available())
print(torch.version.hip)
print(torchvision.__version__)
print(torch.__version__)
print(torch_directml.device_count())