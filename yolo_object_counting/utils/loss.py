import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=2.0, lambda_noobj=1.0):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')

    def forward(self, predictions, targets):
        total_loss = 0
        for scale_idx, (pred, target) in enumerate(zip(predictions, targets)):
            B, H, W, A, _ = pred.shape  # [batch, height, width, anchors, 6]
            
            # Rozdziel predykcje
            pred_xy = pred[..., 0:2]  # x, y
            pred_wh = pred[..., 2:4]  # w, h
            pred_conf = torch.sigmoid(pred[..., 4])  # confidence
            pred_cls = torch.sigmoid(pred[..., 5])  # class score
            
            # Rozdziel targety
            target_xy = target[..., 0:2]
            target_wh = target[..., 2:4]
            target_conf = target[..., 4]
            target_cls = target[..., 5]
            
            # Maska dla obiektów i braku obiektów
            obj_mask = target_conf > 0
            noobj_mask = target_conf == 0
            obj_count = obj_mask.sum().item()
            noobj_count = noobj_mask.sum().item()
            
            print(f"Scale {scale_idx}: Predictions shape: {pred.shape}, Targets shape: {target.shape}")
            print(f"Scale {scale_idx} - Obiekty: {obj_count}, Brak obiektów: {noobj_count}")
            
            # Strata dla współrzędnych (tylko dla obiektów)
            if obj_count > 0:
                loss_xy = self.mse(pred_xy[obj_mask], target_xy[obj_mask])
                loss_wh = self.mse(pred_wh[obj_mask], target_wh[obj_mask])
                loss_coord = self.lambda_coord * (loss_xy + loss_wh)
            else:
                loss_coord = torch.tensor(0.0, device=pred.device)
            
            # Strata dla ufności (confidence)
            loss_conf = self.bce(pred_conf[obj_mask], target_conf[obj_mask]) if obj_count > 0 else torch.tensor(0.0, device=pred.device)
            loss_noobj = self.bce(pred_conf[noobj_mask], target_conf[noobj_mask]) if noobj_count > 0 else torch.tensor(0.0, device=pred.device)
            loss_conf = loss_conf + self.lambda_noobj * loss_noobj
            
            # Strata dla klas (tylko dla obiektów)
            loss_cls = self.bce(pred_cls[obj_mask], target_cls[obj_mask]) if obj_count > 0 else torch.tensor(0.0, device=pred.device)
            
            # Całkowita strata dla tej skali
            scale_loss = loss_coord + loss_conf + loss_cls
            total_loss += scale_loss
        
        return total_loss / B