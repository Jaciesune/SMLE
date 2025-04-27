# backend/api/test_distilled.py

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF
from tqdm import tqdm
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18

# Ścieżki
DISTILLED_DIR = "backend/distilled_models"
TEST_IMAGES_DIR = "backend/data/test/images/"
TEST_ANNOTATIONS_PATH = "backend/data/test/annotations/instances_test.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CocoTestDataset(CocoDetection):
    def __init__(self, img_folder, ann_file):
        super(CocoTestDataset, self).__init__(img_folder, ann_file)
    
    def __getitem__(self, idx):
        img, target = super(CocoTestDataset, self).__getitem__(idx)
        img = TF.to_tensor(img)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def create_student_model():
    backbone = resnet18(weights="IMAGENET1K_V1")
    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
        in_channels_list=[64, 128, 256, 512],
        out_channels=256
    )
    model = MaskRCNN(backbone_with_fpn, num_classes=2)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader):
    model.eval()
    total_images = 0
    total_detections = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Testowanie"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for output in outputs:
                total_detections += len(output["boxes"])
            total_images += len(images)

    print(f"\nŚrednia liczba wykryć na obraz: {total_detections / total_images:.2f}")

def list_distilled_models():
    return [f for f in os.listdir(DISTILLED_DIR) if f.endswith(".pth")]

def run_test(selected_model_name):
    model_path = os.path.join(DISTILLED_DIR, selected_model_name)
    print(f"Ładowanie destylatu: {model_path}")

    model = create_student_model()
    model.load_state_dict(torch.load(model_path, map_location=device))

    dataset = CocoTestDataset(TEST_IMAGES_DIR, TEST_ANNOTATIONS_PATH)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    print("\nRozpoczynam testowanie...")
    evaluate_model(model, dataloader)

if __name__ == "__main__":
    models = list_distilled_models()
    print("Dostępne destylaty do testu:")
    for idx, model_name in enumerate(models):
        print(f"[{idx}] {model_name}")

    choice = int(input("Wybierz destylat do testu: "))
    selected_model = models[choice]

    run_test(selected_model)
