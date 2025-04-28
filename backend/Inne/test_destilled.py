import os
import torch
import torchvision
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF
from tqdm import tqdm
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18

# Ścieżki
VISUALS_DIR = "backend/val_distilled_visuals"
DISTILLED_DIR = "backend/distilled_models"
TEST_IMAGES_DIR = "backend/data/val/images/"
TEST_ANNOTATIONS_PATH = "backend/data/val/annotations/instances_val.json"

os.makedirs(VISUALS_DIR, exist_ok=True)

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
        for idx, (images, _) in enumerate(tqdm(dataloader, desc="Testowanie")):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                img = images[i].cpu()
                img = F.to_pil_image(img)

                fig, ax = plt.subplots(1)
                ax.imshow(img)

                boxes = output["boxes"].cpu().numpy()
                for box in boxes:
                    x_min, y_min, width, height = box
                    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                save_path = os.path.join(VISUALS_DIR, f"test_image_{total_images+i}.png")
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

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
