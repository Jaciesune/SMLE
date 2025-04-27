import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import ANCHOR_SIZES, ANCHOR_RATIOS, USE_CUSTOM_ANCHORS, NUM_CLASSES, NMS_THRESHOLD

def get_model(num_classes, device):
    if USE_CUSTOM_ANCHORS:
        anchor_generator = AnchorGenerator(
            sizes=ANCHOR_SIZES,
            aspect_ratios=ANCHOR_RATIOS
        )
    else:
        anchor_generator = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        rpn_anchor_generator=anchor_generator,
        box_nms_thresh=NMS_THRESHOLD,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_batch_size_per_image=256,
        box_detections_per_img=2000,
        box_score_thresh=0.25
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    print(f"Model działa na: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    return model