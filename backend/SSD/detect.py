import os
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import glob

# Ścieżka do obrazów testowych
img_path = "./dataset/test/images"
img_files = sorted(glob.glob(os.path.join(img_path, "*.*")))

# Wybór modelu
models = sorted(glob.glob("saved_models/*.pth"))
print("Dostępne modele:")
for i, model_path in enumerate(models):
    print(f"[{i}] {model_path}")

choice = int(input("Podaj numer modelu do załadowania: "))
model_path = models[choice]

# Ładowanie modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
model.head.classification_head.num_classes = 2
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === Testowanie obrazów ===
os.makedirs("results", exist_ok=True)
transform = torchvision.transforms.v2.ToTensor()

for img_file in img_files:
    img = read_image(img_file).float() / 255.0
    img = transform(img).to(device)

    with torch.no_grad():
        pred = model([img])[0]

    boxes = pred['boxes'][pred['scores'] > 0.5].cpu()
    labels = pred['labels'][pred['scores'] > 0.5].cpu()
    img_drawn = draw_bounding_boxes((img.cpu() * 255).to(torch.uint8), boxes, labels=[str(l.item()) for l in labels], width=2)

    out_path = os.path.join("results", os.path.basename(img_file))
    torchvision.utils.save_image(img_drawn.float() / 255, out_path)

print("Wyniki zapisane w folderze: results/")