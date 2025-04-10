import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from config import (
    ANCHOR_SIZES,
    ANCHOR_RATIOS,
    NMS_THRESHOLD,
    NUM_CLASSES,
    USE_CUSTOM_ANCHORS
)

def get_model(num_classes, device):
    # 1. Backbone
    backbone = resnet_fpn_backbone(
        'resnet50',
        weights=ResNet50_Weights.IMAGENET1K_V1
    )

    # 2. Anchory (jeśli aktywne w config)
    anchor_generator = AnchorGenerator(
        sizes=ANCHOR_SIZES,
        aspect_ratios=ANCHOR_RATIOS
    ) if USE_CUSTOM_ANCHORS else None

    # 3. Tworzymy model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=None,  # <- tutaj ignorujemy
        rpn_anchor_generator=anchor_generator,
        box_nms_thresh=NMS_THRESHOLD
    )

    # 4. Nadpisujemy klasyfikator (tu właśnie używamy num_classes!)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.to(device)
    print(f"Model dziala na: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    return model
