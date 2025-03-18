import cv2
import torch
import numpy as np
import os

def preprocess_image(img, img_size=416):
    """Przetwarza obraz do formatu YOLO."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))

    print(f"Image shape before normalization: {image.shape}")

    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # ✅ Upewniamy się, że to `float32`

    print(f"Processed image shape: {image.shape}, dtype: {image.dtype}")

    return image

def convert_to_xyxy(predictions):
    """Konwertuje bboxy z (x_center, y_center, w, h) do (x1, y1, x2, y2)"""
    print(f"\n🔍 Raw predictions shape: {predictions.shape}")
    print(f"🔍 Sample raw predictions (przed konwersją):\n{predictions[:5]}")  

    x_center, y_center, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]

    min_size = 1e-3  # Minimalna szerokość i wysokość bboxa

    # ✅ Poprawna konwersja bboxów
    x1 = x_center - (w / 2)
    y1 = y_center - (h / 2)
    x2 = x_center + (w / 2)
    y2 = y_center + (h / 2)

    # ✅ Zapewnienie minimalnych rozmiarów bboxa
    x2 = torch.max(x2, x1 + min_size)
    y2 = torch.max(y2, y1 + min_size)

    # ✅ Clamping do zakresu obrazu [0,1]
    x1 = torch.clamp(x1, min=0, max=1)
    y1 = torch.clamp(y1, min=0, max=1)
    x2 = torch.clamp(x2, min=0, max=1)
    y2 = torch.clamp(y2, min=0, max=1)

    converted_preds = torch.stack([x1, y1, x2, y2, predictions[:, 4], predictions[:, 5]], dim=1)

    print(f"✅ Converted predictions (po konwersji):\n{converted_preds[:5]}\n")  
    return converted_preds

def draw_bounding_boxes(orig_img, detections):
    """Rysuje wykryte bboxy na obrazie."""
    h, w, _ = orig_img.shape
    for det in detections:
        if det is None or len(det) == 0:
            continue
        for detection in det:
            x1, y1, x2, y2 = map(float, detection[:4])
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf = detection[4].item()
            cv2.putText(orig_img, f"Conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detect_images_in_folder(folder_path, model, weights_path, conf_threshold=0.7, iou_threshold=0.4, save_results=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device).float()
    model.eval()

    output_dir = "runs/detect/"
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading image {image_path}")
                continue

            orig_img = img.copy()
            input_img = preprocess_image(img).to(device)

            with torch.no_grad():
                predictions = model(input_img)
                B, H, W, num_anchors, _ = predictions.shape
                predictions = predictions.view(B, -1, 6)[0]

                confidences = torch.sigmoid(predictions[:, 4])  # Poprawna normalizacja
                print(f"Image: {image_name}, Confidence values (top 10): {confidences.topk(10).values}")

                predictions = convert_to_xyxy(predictions)
                detections = non_max_suppression([predictions], conf_threshold, iou_threshold)

                print(f"Before NMS: {len(predictions)} boxes, After NMS: {len(detections[0]) if detections[0] is not None else 0} boxes")

                if save_results:
                    draw_bounding_boxes(orig_img, detections)
                    result_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(result_path, orig_img)
                    print(f"Saved result to {result_path}")

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

if __name__ == "__main__":
    from models.yolo import YOLO
    from utils.non_max_suppression import non_max_suppression  
    
    model = YOLO(num_classes=1, num_anchors=3)
    detect_images_in_folder("data/images/test/", model, "yolo_model.pth")
