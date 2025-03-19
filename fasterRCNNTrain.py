import torch
import torchvision.models.detection
import torch.optim as optim
from dataLoader import get_data_loaders  # Importujemy funkcję zamiast loaderów

# Ustawienie wielowątkowości CPU
torch.set_num_threads(8)  # Liczba wątków
torch.set_num_interop_threads(8)

# Ustawienie CPU jako urządzenia
device = torch.device("cpu")

# Pobranie modelu
def get_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    print(f"Model działa na: {device}")
    return model

# Pobranie DataLoaderów
train_loader, test_loader = get_data_loaders(num_workers=4)  # Optymalizacja

# Inicjalizacja modelu i optymalizatora
model = get_model(num_classes=2, device=device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Funkcja treningu
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Trening modelu
num_epochs = 10
for epoch in range(num_epochs):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoka {epoch+1}/{num_epochs}, Strata: {loss:.4f}")

print("Trening zakończony!")