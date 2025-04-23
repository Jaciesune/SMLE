import os
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from tqdm import tqdm
from data_loader import COCODataset

transform = T.ToTensor()
ann_path = "./dataset/train/annotations/instances_train.json"
img_path = "./dataset/train/images"
dataset = COCODataset(img_path, ann_path, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssd300_vgg16(weights="DEFAULT")
model.head.classification_head.num_classes = 2
model = model.to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    for images, targets, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"[Epoch {epoch+1}] Loss: {losses.item():.4f}")

os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/ssd_pipe.pth")