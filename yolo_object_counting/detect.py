import os
import cv2
import torch
import torch.nn as nn
import time
from models.yolo import YOLO
from utils.non_max_suppression import soft_nms

def preprocess_image(img, input_size=(416, 416)):
    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0)

def convert_to_xyxy(prediction, grid_h, grid_w, anchors):
    num_boxes = prediction.shape[0]
    num_anchors = len(anchors)

    print(f"Prediction shape: {prediction.shape}")
    print(f"Grid size: {grid_h}x{grid_w}, num_anchors: {num_anchors}")

    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    grid_x = grid_x.flatten().to(prediction.device)
    grid_y = grid_y.flatten().to(prediction.device)
    print(f"Grid_x shape after flatten: {grid_x.shape}")

    grid_x = grid_x.repeat_interleave(num_anchors)
    grid_y = grid_y.repeat_interleave(num_anchors)
    print(f"Grid_x shape after repeat_interleave: {grid_x.shape}")

    anchors = anchors.repeat(grid_h * grid_w, 1).to(prediction.device)
    print(f"Anchors shape after repeat: {anchors.shape}")

    prediction[..., 0] = torch.sigmoid(prediction[..., 0])
    prediction[..., 1] = torch.sigmoid(prediction[..., 1])
    prediction[..., 2] = torch.exp(prediction[..., 2]) * anchors[:, 0]
    prediction[..., 3] = torch.exp(prediction[..., 3]) * anchors[:, 1]
    prediction[..., 4] = torch.sigmoid(prediction[..., 4])
    prediction[..., 5] = torch.sigmoid(prediction[..., 5])

    prediction[..., 0] = (prediction[..., 0] + grid_x) / grid_w
    prediction[..., 1] = (prediction[..., 1] + grid_y) / grid_h

    prediction[..., 0] = prediction[..., 0] - prediction[..., 2] / 2
    prediction[..., 1] = prediction[..., 1] - prediction[..., 3] / 2
    prediction[..., 2] = prediction[..., 0] + prediction[..., 2]
    prediction[..., 3] = prediction[..., 1] + prediction[..., 3]

    prediction[..., 0] = torch.clamp(prediction[..., 0], 0, 1)
    prediction[..., 1] = torch.clamp(prediction[..., 1], 0, 1)
    prediction[..., 2] = torch.clamp(prediction[..., 2], 0, 1)
    prediction[..., 3] = torch.clamp(prediction[..., 3], 0, 1)

    print(f"Sample widths (scale {grid_h}x{grid_w}): {prediction[:5, 2]}")
    print(f"Sample heights (scale {grid_h}x{grid_w}): {prediction[:5, 3]}")
    print(f"Sample bounding boxes (scale {grid_h}x{grid_w}): {prediction[:5, :4]}")

    return prediction

def draw_bounding_boxes(img, detections):
    if detections is None or len(detections) == 0:
        return

    detections = detections.cpu().numpy()
    h, w = img.shape[:2]

    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"Pipe: {conf:.2f}"
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detect_images_in_folder(folder_path, model, weights_path, conf_threshold=0.3, iou_threshold=0.6, sigma=0.5, save_results=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device).float()
    model.eval()

    output_dir = "runs/detect/"
    os.makedirs(output_dir, exist_ok=True)

    grid_sizes = [104, 52, 26, 13]

    anchors = torch.tensor([
        [0.03, 0.05],
        [0.05, 0.08],
        [0.07, 0.10],
        [0.02, 0.03],
        [0.04, 0.06],
        [0.08, 0.12],
        [0.06, 0.09],
        [0.10, 0.15]
    ])

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    total_start_time = time.time()

    all_images = []
    all_orig_images = []

    preprocess_start = time.time()
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}")
            continue
        orig_img = img.copy()
        input_img = preprocess_image(img)
        all_images.append(input_img)
        all_orig_images.append((orig_img, image_name))
    print(f"Preprocessing time for all images: {time.time() - preprocess_start:.3f} seconds")

    if all_images:
        input_batch = torch.cat(all_images, dim=0).to(device)

        with torch.no_grad():
            forward_start = time.time()
            predictions = model(input_batch)
            print(f"Forward pass time for all images: {time.time() - forward_start:.3f} seconds")

            for batch_idx, (orig_img, image_name) in enumerate(all_orig_images):
                start_time = time.time()
                all_predictions = []
                convert_start = time.time()
                for scale_idx, pred in enumerate(predictions):
                    pred_single = pred[batch_idx]
                    H, W, num_anchors, _ = pred_single.shape
                    pred_single = pred_single.view(-1, 6)
                    confidences = torch.sigmoid(pred_single[:, 4])
                    print(f"Image {image_name}, Scale {scale_idx} - Confidence values (top 10): {confidences.topk(10).values}")
                    pred_single = convert_to_xyxy(pred_single, grid_h=grid_sizes[scale_idx], grid_w=grid_sizes[scale_idx], anchors=anchors)
                    all_predictions.append(pred_single)
                print(f"Convert to xyxy time for {image_name}: {time.time() - convert_start:.3f} seconds")

                all_predictions = torch.cat(all_predictions, dim=0)

                filter_start = time.time()
                conf_threshold_pre_nms = 0.95
                mask = all_predictions[:, 4] > conf_threshold_pre_nms
                all_predictions = all_predictions[mask]
                print(f"Image {image_name}, After pre-NMS filtering: {len(all_predictions)} boxes")
                print(f"Filtering time for {image_name}: {time.time() - filter_start:.3f} seconds")

                nms_start = time.time()
                initial_detections = soft_nms(all_predictions, conf_threshold=0.6, iou_threshold=0.7, sigma=sigma, use_standard_nms=True)
                num_detections = len(initial_detections) if len(initial_detections) > 0 else 0
                print(f"Initial NMS time for {image_name}: {time.time() - nms_start:.3f} seconds")

                if num_detections > 50:
                    conf_threshold = 0.9
                    iou_threshold = 0.7
                    sigma = 0.5
                else:
                    conf_threshold = 0.9
                    iou_threshold = 0.6
                    sigma = 0.5

                print(f"Image {image_name}, Using conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, sigma={sigma} for {num_detections} initial detections")
                nms_start = time.time()
                detections = soft_nms(all_predictions, conf_threshold=conf_threshold, iou_threshold=iou_threshold, sigma=sigma, use_standard_nms=True)
                print(f"Final NMS time for {image_name}: {time.time() - nms_start:.3f} seconds")

                print(f"Image {image_name}, Before NMS: {len(all_predictions)} boxes, After NMS: {len(detections) if len(detections) > 0 else 0} boxes")

                if len(detections) > 100:
                    _, indices = torch.sort(detections[:, 4], descending=True)
                    detections = detections[indices[:100]]

                if save_results:
                    draw_start = time.time()
                    draw_bounding_boxes(orig_img, detections)
                    result_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(result_path, orig_img)
                    print(f"Draw and save time for {image_name}: {time.time() - draw_start:.3f} seconds")

                print(f"Total time for {image_name}: {time.time() - start_time:.3f} seconds")

    print(f"Total time for all images: {time.time() - total_start_time:.3f} seconds")

if __name__ == "__main__":
    from models.yolo import YOLO
    from utils.non_max_suppression import soft_nms

    model = YOLO(num_classes=1, num_anchors=8)
    detect_images_in_folder("data/images/test/", model, "yolo_model.pth")