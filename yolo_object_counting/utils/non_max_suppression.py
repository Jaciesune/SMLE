import torch

def soft_nms(predictions, conf_threshold=0.3, iou_threshold=0.6, sigma=0.5):
    """
    Wykonuje Soft-NMS na predykcjach modelu YOLO.
    
    Args:
        predictions (tensor): Tensor z predykcjami o kształcie [num_boxes, 6]
                             (x_min, y_min, x_max, y_max, confidence, class_score).
        conf_threshold (float): Próg ufności dla filtrowania predykcji.
        iou_threshold (float): Próg IoU dla Soft-NMS.
        sigma (float): Parametr kontrolujący miękkie tłumienie.
    
    Returns:
        Tensor: Przefiltrowane predykcje po Soft-NMS.
    """
    if len(predictions) == 0:
        return torch.tensor([], device=predictions.device)

    # Filtruj predykcje na podstawie progu ufności
    confidences = predictions[:, 4]
    mask = confidences >= conf_threshold
    predictions = predictions[mask]
    if len(predictions) == 0:
        return torch.tensor([], device=predictions.device)

    # Sortuj predykcje według ufności (malejąco)
    _, indices = torch.sort(confidences[mask], descending=True)
    predictions = predictions[indices]

    keep = []
    while len(predictions) > 0:
        # Weź predykcję z najwyższą ufnością
        keep.append(predictions[0])
        if len(predictions) == 1:
            break

        # Oblicz IoU między wybraną predykcją a pozostałymi
        ious = torch.tensor([calculate_iou(predictions[0, :4], pred[:4]) for pred in predictions[1:]], device=predictions.device)

        # Obniż ufność predykcji, które mają wysokie IoU
        weights = torch.exp(-(ious ** 2) / sigma)
        predictions[1:, 4] = predictions[1:, 4] * weights

        # Usuń predykcje z ufnością poniżej progu
        mask = predictions[1:, 4] >= conf_threshold
        predictions = predictions[1:][mask]

    return torch.stack(keep) if keep else torch.tensor([], device=predictions.device)

def calculate_iou(box1, box2):
    """
    Oblicza IoU (Intersection over Union) między dwoma bounding boxami.
    
    Args:
        box1 (tensor): Pierwszy bounding box [x_min, y_min, x_max, y_max].
        box2 (tensor): Drugi bounding box [x_min, y_min, x_max, y_max].
    
    Returns:
        float: Wartość IoU.
    """
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)