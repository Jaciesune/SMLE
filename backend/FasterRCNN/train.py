import torch
import sys

sys.stdout.reconfigure(encoding='utf-8')

def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for images, targets in train_loader:
        if images is None or targets is None:
            print("Ostrzeżenie: pominięto partię z powodu None w obrazach lub targetach.")
            continue

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Konwersja i normalizacja obrazów tutaj, bezpośrednio przed przekazaniem do modelu
        images = [(image.float() / 255.0) for image in images]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0