import torch
import torch.nn as nn

def box_iou(box1, box2):
    """
    Oblicza IoU pomiędzy dwoma zestawami bounding boxów.
    box1: (N, 4) - (x1, y1, x2, y2)
    box2: (M, 4) - (x1, y1, x2, y2)
    """
    # Konwersja z (center_x, center_y, width, height) na (x1, y1, x2, y2)
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2
    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2
    
    # Koordynaty obszaru przecięcia
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # Obliczanie powierzchni przecięcia
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Obliczanie powierzchni bounding boxów
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # Obliczanie IoU
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    
    return iou


class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # Debug: Sprawdź wartości wejściowe
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("Warning: NaN or Inf detected in predictions!")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN or Inf detected in targets!")

        # Object and no-object masks
        obj_mask = targets[..., 4] == 1
        threshold = 0.3
        # Uproszczona maska noobj, bez warunku na predykcje
        noobj_mask = targets[..., 4] == 0

        # Liczba obiektów i brak obiektów
        obj_count = obj_mask.sum().item()
        noobj_count = noobj_mask.sum().item()
        print(f"Obiekty: {obj_count}, Brak obiektów: {noobj_count}")

        # Confidence loss
        pred_conf = predictions[..., 4]
        target_conf = targets[..., 4]
        loss_conf = self.bce(pred_conf[obj_mask], target_conf[obj_mask]) if obj_count > 0 else torch.tensor(0.0)

        # No object loss (obsługa przypadku pustej maski)
        loss_noobj = self.bce(pred_conf[noobj_mask], target_conf[noobj_mask]) if noobj_count > 0 else torch.tensor(0.0)

        # Classification loss
        pred_cls = predictions[..., 5:]
        target_cls = targets[..., 5:]
        loss_class = self.bce(pred_cls[obj_mask], target_cls[obj_mask]) if obj_count > 0 else torch.tensor(0.0)

        # Localization loss
        pred_boxes = predictions[..., :4]
        target_boxes = targets[..., :4]
        pred_boxes_wh = torch.sqrt(torch.abs(pred_boxes[..., 2:4]) + 1e-6)
        target_boxes_wh = torch.sqrt(target_boxes[..., 2:4] + 1e-6)

        loss_coord = self.mse(
            torch.cat((pred_boxes[..., :2], pred_boxes_wh), dim=-1)[obj_mask],
            torch.cat((target_boxes[..., :2], target_boxes_wh), dim=-1)[obj_mask]
        ) * self.lambda_coord if obj_count > 0 else torch.tensor(0.0)

        # Total loss
        loss = loss_coord + loss_conf + self.lambda_noobj * loss_noobj + loss_class
        print(f"Loss breakdown: Coord={loss_coord.item() if obj_count > 0 else 0.0:.2f}, Conf={loss_conf.item() if obj_count > 0 else 0.0:.2f}, NoObj={loss_noobj.item() if noobj_count > 0 else 0.0:.2f}, Class={loss_class.item() if obj_count > 0 else 0.0:.2f}")
        return loss