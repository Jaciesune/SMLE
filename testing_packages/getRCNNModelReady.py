import torch
import torchvision.models.detection
import torch_directml

# Pobranie modelu Faster R-CNN z pretrenowanymi wagami
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Dostosowanie modelu do klasy ("rura" + tło)
num_classes = 2  # 1 klasa ("rura") + tło
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Przeniesienie modelu na DirectML
device = torch_directml.device()
model.to(device)

print("Model Faster R-CNN gotowy do treningu na:", device)