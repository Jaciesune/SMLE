# backend/api/distillation.py

import os
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import functional as TF
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn_v2

# Ścieżki
MODELS_DIR = "backend/Mask_RCNN/models"
DISTILLED_DIR = "backend/distilled_models"
TRAIN_IMAGES_DIR = "backend/data/train/images/"
TRAIN_ANNOTATIONS_PATH = "backend/data/train/annotations/instances_train.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_available_models():
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]

def load_teacher_model(model_path):
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=2
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def create_student_model():
    backbone = resnet18(weights="IMAGENET1K_V1")
    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
        in_channels_list=[64, 128, 256, 512],
        out_channels=256
    )
    model = MaskRCNN(backbone_with_fpn, num_classes=2)
    model.roi_heads.detections_per_img = 1000
    model.roi_heads.nms_thresh = 0.15
    model.roi_heads.score_thresh = 0.5
    model.to(device)
    return model

class CocoTrainDataset(CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)
    
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = TF.to_tensor(img)
        target = self.prepare_target(target)
        return img, target

    def prepare_target(self, target):
        boxes, labels, masks = [], [], []
        for obj in target:
            x, y, width, height = obj["bbox"]
            x_min = x
            y_min = y
            x_max = x + width
            y_max = y + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj["category_id"])
            
            # Tworzymy sztuczną maskę 10x10
            masks.append(torch.zeros((10, 10), dtype=torch.uint8))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, 10, 10), dtype=torch.uint8)

        return {"boxes": boxes, "labels": labels, "masks": masks}

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def distillation_loss(student_outputs, teacher_outputs):
    losses = []
    for s, t in zip(student_outputs, teacher_outputs):
        num_boxes = min(s["boxes"].shape[0], t["boxes"].shape[0])

        if num_boxes == 0:
            continue

        losses.append(F.l1_loss(s["boxes"][:num_boxes], t["boxes"][:num_boxes]))

        if "scores" in s and "scores" in t:
            losses.append(F.l1_loss(s["scores"][:num_boxes], t["scores"][:num_boxes]))

        if "masks" in s and "masks" in t:
            mask_num = min(s["masks"].shape[0], t["masks"].shape[0])
            losses.append(F.l1_loss(s["masks"][:mask_num], t["masks"][:mask_num]))

    if losses:
        return sum(losses) / len(losses)
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)

def distill_model(teacher, student, dataloader, epochs=60):
    optimizer = torch.optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for images, targets in tqdm(dataloader, desc=f"Destylacja Epoch {epoch+1}"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                teacher_outputs = teacher(images)

            student.train()
            with torch.set_grad_enabled(True):
                detector_losses = student(images, targets)  # <-- tylko jedna zmienna!
                loss_detection = sum(loss for loss in detector_losses.values())

            student.eval()
            with torch.no_grad():
                student_outputs = student(images)

            loss_distill = distillation_loss(student_outputs, teacher_outputs)

            loss = loss_detection + 0.5 * loss_distill  # 0.5 = waga destylacji

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            torch.cuda.empty_cache()

        print(f"Loss: {running_loss / len(dataloader):.4f}")

def save_student_model(student, save_path):
    torch.save(student.state_dict(), save_path)

def run_distillation(selected_model_name):
    teacher_model_path = os.path.join(MODELS_DIR, selected_model_name)
    print(f"Ładowanie modelu Teacher: {teacher_model_path}")

    teacher = load_teacher_model(teacher_model_path)
    student = create_student_model()

    dataset = CocoTrainDataset(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    print("Rozpoczynam destylację...")
    distill_model(teacher, student, dataloader, epochs=20)

    distilled_model_name = selected_model_name.replace(".pth", "_distilled.pth")
    distilled_model_path = os.path.join(DISTILLED_DIR, distilled_model_name)

    os.makedirs(DISTILLED_DIR, exist_ok=True)
    save_student_model(student, distilled_model_path)
    print(f"Destylat zapisany: {distilled_model_path}")

if __name__ == "__main__":
    models = list_available_models()
    print("Dostępne modele do destylacji:")
    for idx, model_name in enumerate(models):
        print(f"[{idx}] {model_name}")

    choice = int(input("Wybierz model do destylacji: "))
    selected_model = models[choice]

    run_distillation(selected_model)
