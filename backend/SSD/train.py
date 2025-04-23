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
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json

transform = T.ToTensor()

train_ann = "./dataset/train/annotations/instances_train.json"
train_img = "./dataset/train/images"
train_dataset = COCODataset(train_img, train_ann, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

val_ann = "./dataset/val/annotations/instances_val.json"
val_img = "./dataset/val/images"
val_dataset = COCODataset(val_img, val_ann, transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

val_coco = COCO(val_ann)

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
train_losses, val_losses, map_scores = [], [], []

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
    coco_results = []

    with torch.no_grad():
        for idx, (images, targets, paths) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} - Val")):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()

            preds = model(images)
            for i, pred in enumerate(preds):
                boxes = pred['boxes'].detach().cpu()
                scores = pred['scores'].detach().cpu()
                labels = pred['labels'].detach().cpu()

                image_id = int(targets[i]['image_id'].item())
                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.05:
                        continue
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                        "score": float(score.item())
                    })

            if idx in val_indices:
                pred = preds[0]
                boxes = pred['boxes'][pred['scores'] > 0.5].cpu()
                labels = pred['labels'][pred['scores'] > 0.5].cpu()
                img_drawn = draw_bounding_boxes((images[0].cpu() * 255).to(torch.uint8), boxes, labels=[str(l.item()) for l in labels], width=2)
                out_path = os.path.join("val_visuals", f"epoch{epoch+1}_img{idx}.jpg")
                save_image(img_drawn.float() / 255, out_path)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

    # --- mAP @ IoU=0.5 ---
    if len(coco_results) > 0:
        with open("temp_results.json", "w") as f:
            json.dump(coco_results, f, indent=2)

        coco_dt = val_coco.loadRes("temp_results.json")
        coco_eval = COCOeval(val_coco, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map50 = coco_eval.stats[1]  # AP@IoU=0.5
        map_scores.append(map50)
        print(f"[Epoch {epoch+1}] mAP@0.5: {map50:.4f}")
    else:
        map_scores.append(0.0)

    # Zapis najlepszego modelu
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "saved_models/best_ssd_pipe.pth")
        print(f"Zapisano najlepszy model z Val Loss: {best_val_loss:.4f}")
        for f in os.listdir("saved_models"):
            if f != "best_ssd_pipe.pth":
                os.remove(os.path.join("saved_models", f))

# Wykresy
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(map_scores, label="mAP@0.5")
plt.xlabel("Epoka")
plt.ylabel("Wartość")
plt.title("Straty i mAP podczas treningu")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.close()