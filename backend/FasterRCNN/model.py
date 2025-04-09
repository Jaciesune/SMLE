import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights
import torch

def get_model(num_classes, device):
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        rpn_anchor_generator=anchor_generator
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.score_thresh = 0.25
    model.roi_heads.nms_thresh = 0.14
    model.roi_heads.detections_per_img = 4000
    model.to(device)

    print(f"Model działa na: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    return model
