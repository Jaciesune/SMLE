import os
import torch
import cv2
import numpy as np
from utils import filter_and_draw_boxes

def validate_model(model, dataloader, device, epoch, model_name):
    model.train()
    total_val_loss = 0
    total_pred_objects = 0
    total_gt_objects = 0
    save_path = f"val/{model_name}/epoch_{epoch:02d}"
    os.makedirs(save_path, exist_ok=True)

    for idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, new_targets)
        val_loss = sum(loss for loss in loss_dict.values())
        total_val_loss += val_loss.item()

        model.eval()
        with torch.no_grad():
            outputs = model(images)

        for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
            image_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            gt_count = target["boxes"].shape[0]
            total_gt_objects += gt_count

            image_np, pred_count = filter_and_draw_boxes(
                image_np,
                output["boxes"].detach().cpu().numpy(),
                output["scores"].detach().cpu().numpy(),
                image_np.shape[:2]
            )
            total_pred_objects += pred_count

            filename = f"{save_path}/img_{idx}_{i}.png"
            cv2.imwrite(filename, image_np)

        model.train()

    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_val_loss, total_pred_objects, total_gt_objects