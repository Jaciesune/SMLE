import torch
from torchvision.ops import nms

def soft_nms(predictions, conf_threshold=0.3, iou_threshold=0.6, sigma=0.5, use_standard_nms=False):
    if len(predictions) == 0:
        return torch.tensor([], device=predictions.device)

    confidences = predictions[:, 4]
    mask = confidences >= conf_threshold
    predictions = predictions[mask]
    if len(predictions) == 0:
        return torch.tensor([], device=predictions.device)

    if use_standard_nms:
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        keep_indices = nms(boxes, scores, iou_threshold)
        return predictions[keep_indices]

    _, indices = torch.sort(confidences[mask], descending=True)
    predictions = predictions[indices]

    keep = []
    while len(predictions) > 0:
        keep.append(predictions[0])
        if len(predictions) == 1:
            break

        ious = calculate_iou(predictions[0, :4], predictions[1:, :4])
        weights = torch.exp(-(ious ** 2) / sigma)
        predictions[1:, 4] = predictions[1:, 4] * weights

        mask = predictions[1:, 4] >= conf_threshold
        predictions = predictions[1:][mask]

    return torch.stack(keep) if keep else torch.tensor([], device=predictions.device)

def calculate_iou(box1, boxes):
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)