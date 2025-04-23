import os
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torch.utils.data import DataLoader
from data_loader import COCODataset

ann_path = "./dataset/train/annotations/instances_train.json"
img_path = "./dataset/train/images"
transform = torchvision.transforms.v2.ToTensor()
dataset = COCODataset(img_path, ann_path, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
model.head.classification_head.num_classes = 2
model.load_state_dict(torch.load("saved_models/ssd_pipe.pth", map_location=device))
model = model.to(device)
model.eval()

os.makedirs("results", exist_ok=True)
for images, _, paths in dataloader:
    img = images[0].to(device)
    with torch.no_grad():
        pred = model([img])[0]

    boxes = pred['boxes'][pred['scores'] > 0.5].cpu()
    labels = pred['labels'][pred['scores'] > 0.5].cpu()
    img_drawn = draw_bounding_boxes((img.cpu() * 255).to(torch.uint8), boxes, labels=[str(l.item()) for l in labels], width=2)

    out_path = os.path.join("results", paths[0])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchvision.utils.save_image(img_drawn.float() / 255, out_path)

print("Wyniki zapisane w folderze: results/")