import os
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.utils import draw_bounding_boxes, save_image
from tqdm import tqdm
from data_loader import COCODataset
import matplotlib.pyplot as plt
import random

transform = T.ToTensor()

train_ann = "./dataset/train/annotations/instances_train.json"
train_img = "./dataset/train/images"
train_dataset = COCODataset(train_img, train_ann, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

val_ann = "./dataset/val/annotations/instances_val.json"
val_img = "./dataset/val/images"
val_dataset = COCODataset(val_img, val_ann, transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Wybierz 10 losowych indeksów z walidacji do wizualizacji
val_indices = random.sample(range(len(val_dataset)), min(10, len(val_dataset)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssd300_vgg16(weights="DEFAULT")
model.head.classification_head.num_classes = 2
model = model.to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

os.makedirs("saved_models", exist_ok=True)
os.makedirs("val_visuals", exist_ok=True)
best_val_loss = float("inf")
train_losses, val_losses = [], []

for epoch in range(10):
    model.train()
    epoch_train_loss = 0
    for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_train_loss += losses.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, (images, targets, paths) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} - Val")):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()

            # Wizualizacja 10 losowych obrazów z walidacji
            global_index = idx  # indeks obrazu w całym zbiorze
            if global_index in val_indices:
                pred = model(images)[0]
                boxes = pred['boxes'][pred['scores'] > 0.5].cpu()
                labels = pred['labels'][pred['scores'] > 0.5].cpu()
                img_drawn = draw_bounding_boxes((images[0].cpu() * 255).to(torch.uint8), boxes, labels=[str(l.item()) for l in labels], width=2)
                out_path = os.path.join("val_visuals", f"epoch{epoch+1}_img{global_index}.jpg")
                save_image(img_drawn.float() / 255, out_path)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

    # Zapis najlepszego modelu
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "saved_models/best_ssd_pipe.pth")
        print(f"Zapisano najlepszy model z Val Loss: {best_val_loss:.4f}")

        # Usuwanie innych modeli
        for f in os.listdir("saved_models"):
            if f != "best_ssd_pipe.pth":
                os.remove(os.path.join("saved_models", f))

# Rysowanie wykresów
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.title("Krzywe strat treningowych i walidacyjnych")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.close()