import os
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# Ścieżki
VISUALS_DIR = "backend/val_distilled_visuals"
DISTILLED_DIR = "backend/distilled_models"
TEST_IMAGES_DIR = "backend/data/val/images/"
TEST_ANNOTATIONS_PATH = "backend/data/val/annotations/instances_val.json"

os.makedirs(VISUALS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CocoTestDataset(CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = TF.to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_student_model_from_arch(arch: dict):
    backbone = resnet18(weights="IMAGENET1K_V1")
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = arch.get("out_channels", 512)

    anchor_generator = AnchorGenerator(
        sizes=arch.get("anchor_sizes", ((32, 64, 128, 256),)),
        aspect_ratios=arch.get("aspect_ratios", ((0.5, 1.0, 2.0),))
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=arch.get("featmap_names", ["0"]),
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    model.roi_heads.detections_per_img = 1000
    model.roi_heads.nms_thresh = 0.15
    model.roi_heads.score_thresh = 0.5

    model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader):
    model.eval()
    total_images = 0
    total_detections = 0

    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(dataloader, desc="Testowanie")):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                img = images[i].cpu()
                img = F.to_pil_image(img)

                fig, ax = plt.subplots(1)
                ax.imshow(img)

                boxes = output["boxes"].cpu().numpy()
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = patches.Rectangle(
                        (x_min, y_min), width, height,
                        linewidth=1, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)

                save_path = os.path.join(VISUALS_DIR, f"test_image_{total_images + i}.png")
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

    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        arch = checkpoint.get("arch", {})
        state_dict = checkpoint["model_state_dict"]
    else:
        # Kompatybilność wsteczna
        arch = {
            "out_channels": 512,
            "anchor_sizes": ((32, 64, 128, 256),),
            "aspect_ratios": ((0.5, 1.0, 2.0),),
            "featmap_names": ["0"]
        }
        state_dict = checkpoint

    model = create_student_model_from_arch(arch)
    model.load_state_dict(state_dict)

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
