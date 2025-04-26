import os
import json
import random
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data_loader import COCODataset

os.environ["TORCH_HOME"] = "./weights"
os.makedirs("./weights/checkpoints", exist_ok=True)
os.makedirs("val_visuals", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()

# Hyperparams
dataset_dir = "./dataset"
lr = 1e-4
score_thresh = 0.01

# Transforms
transform = Compose([
    Resize((300, 300)),
    ToTensor(),
    Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.229, 0.224, 0.225])
])

def make_loader(split):
    ds = COCODataset(
        f"{dataset_dir}/{split}/images",
        f"{dataset_dir}/{split}/annotations/instances_{split}.json",
        transform
    )
    bs = 1 if split == "val" else args.batch_size
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == "train"),
        collate_fn=lambda b: tuple(zip(*b))
    )

train_loader = make_loader("train")
val_loader = make_loader("val")
val_coco = COCO(f"{dataset_dir}/val/annotations/instances_val.json")
val_indices = random.sample(range(len(val_loader.dataset)), min(10, len(val_loader.dataset)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ssd300_vgg16(weights=None, weights_backbone=VGG16_Weights.DEFAULT, num_classes=2)
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_val_loss = float("inf")
hist = {"train": [], "val": [], "map50": []}

for epoch in range(1, args.epochs + 1):
    model.train()
    total_train = 0.0
    for imgs, tgts, _ in tqdm(train_loader, desc=f"Train {epoch}"):
        imgs = [i.to(device) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        if isinstance(loss_dict, list):
            loss_dict = loss_dict[0]
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train += loss.item()

    avg_train = total_train / len(train_loader)
    hist["train"].append(avg_train)
    print(f"Epoch {epoch} Train Loss: {avg_train:.4f}")

    model.eval()
    results = []
    with torch.no_grad():
        for idx, (imgs, tgts, _) in enumerate(tqdm(val_loader, desc=f"Val {epoch}")):
            for j in range(len(imgs)):
                img = imgs[j].to(device).unsqueeze(0)
                tgt = {k: v[j].unsqueeze(0).to(device) if v.dim() > 1 else v[j].to(device) for k, v in tgts[j].items()}

                try:
                    pred = model(img)[0]
                except Exception as e:
                    print("Val error:", e)
                    continue

                image_id = int(tgt["image_id"].item())

                for b, s, l in zip(pred["boxes"].cpu(), pred["scores"].cpu(), pred["labels"].cpu()):
                    if s < score_thresh:
                        continue
                    results.append({
                        "image_id": image_id,
                        "category_id": int(l.item()),
                        "bbox": [b[0].item(), b[1].item(), (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                        "score": s.item()
                    })

    # Nie obliczamy już straty walidacyjnej
    hist["val"].append(0.0)
    print(f"Epoch {epoch} Val Loss: 0.0000")

    if results:
        with open("tmp_results.json", "w") as f:
            json.dump(results, f)
        coco_dt = val_coco.loadRes("tmp_results.json")
        ev = COCOeval(val_coco, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        map50 = ev.stats[1]
    else:
        map50 = 0.0

    hist["map50"].append(map50)
    print(f"Epoch {epoch} mAP@0.5: {map50:.4f}")

    if hist["val"][-1] < best_val_loss:
        best_val_loss = hist["val"][-1]
        torch.save(model.state_dict(), "saved_models/best.pth")

plt.plot(hist["train"], label="train")
plt.plot(hist["val"], label="val")
plt.plot(hist["map50"], label="mAP@0.5")
plt.legend(); plt.grid()
plt.savefig("training_plot.png")
