import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

# Ścieżki
MODELS_DIR = "backend/FasterRCNN/saved_models"
DISTILLED_DIR = "backend/distilled_models"
TRAIN_IMAGES_DIR = "backend/data/train/images/"
TRAIN_ANNOTATIONS_PATH = "backend/data/train/annotations/instances_train.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_available_models():
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]


def load_teacher_model(model_path):
    model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def create_student_model():
    backbone = resnet18(weights="IMAGENET1K_V1")
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

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
        boxes, labels = [], []
        for obj in target:
            x, y, width, height = obj["bbox"]
            x_min = x
            y_min = y
            x_max = x + width
            y_max = y + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj["category_id"])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {"boxes": boxes, "labels": labels}


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

    if losses:
        return sum(losses) / len(losses)
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def distill_model(teacher, student, dataloader, epochs=20):
    optimizer = torch.optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for images, targets in tqdm(dataloader, desc=f"Destylacja Epoch {epoch+1}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                teacher_outputs = teacher(images)

            student.train()
            with torch.set_grad_enabled(True):
                detector_losses = student(images, targets)
                loss_detection = sum(loss for loss in detector_losses.values())

            student.eval()
            with torch.no_grad():
                student_outputs = student(images)

            loss_distill = distillation_loss(student_outputs, teacher_outputs)
            loss = loss_detection + 0.5 * loss_distill

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
    distill_model(teacher, student, dataloader, epochs=1)

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
