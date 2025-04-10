import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import (
    ANCHOR_SIZES,
    ANCHOR_RATIOS,
    NMS_THRESHOLD,
    NUM_CLASSES,
    USE_CUSTOM_ANCHORS
)

def get_model(num_classes, device):
    # Backbone z FPN i pretrenowanymi wagami
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V1)

    # Jeśli używamy custom anchorów – twórz generator
    if USE_CUSTOM_ANCHORS:
        anchor_generator = AnchorGenerator(
            sizes=ANCHOR_SIZES,
            aspect_ratios=ANCHOR_RATIOS
        )
        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_nms_thresh=NMS_THRESHOLD
        )
    else:
        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            box_nms_thresh=NMS_THRESHOLD
        )

    # Nadpisujemy box_predictora (dla pewności, np. przy transfer learningu)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    print(f"Model działa na: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    return model