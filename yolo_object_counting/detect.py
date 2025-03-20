import os
import cv2
import torch
import torch.nn as nn
from models.yolo import YOLO
from utils.non_max_suppression import soft_nms

def preprocess_image(img, input_size=(416, 416)):
    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0)

def convert_to_xyxy(prediction, grid_h, grid_w, anchors):
    """
    Konwertuje predykcje z formatu (x_center, y_center, w, h) na (x_min, y_min, x_max, y_max).
    
    Args:
        prediction (tensor): Tensor o kształcie [num_boxes, 6] (po spłaszczeniu).
        grid_h (int): Wysokość siatki.
        grid_w (int): Szerokość siatki.
        anchors (tensor): Tensor o kształcie [num_anchors, 2] z anchorami (w, h).
    
    Returns:
        Tensor: Przekształcone predykcje w formacie [x_min, y_min, x_max, y_max, conf, cls].
    """
    num_boxes = prediction.shape[0]
    num_anchors = len(anchors)

    # Oblicz pozycje komórek siatki
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    grid_x = grid_x.flatten().to(prediction.device)  # [H * W]
    grid_y = grid_y.flatten().to(prediction.device)  # [H * W]

    # Powtórz grid_x i grid_y, aby pasowały do liczby anchorów
    grid_x = grid_x.repeat_interleave(num_anchors)  # [H * W * num_anchors]
    grid_y = grid_y.repeat_interleave(num_anchors)  # [H * W * num_anchors]

    # Powtórz anchory, aby pasowały do liczby komórek
    anchors = anchors.repeat(grid_h * grid_w, 1).to(prediction.device)  # [H * W * num_anchors, 2]

    # Przekształć predykcje
    prediction[..., 0] = torch.sigmoid(prediction[..., 0])  # x_center
    prediction[..., 1] = torch.sigmoid(prediction[..., 1])  # y_center
    # Skaluj w i h względem anchorów
    prediction[..., 2] = torch.exp(prediction[..., 2]) * anchors[:, 0]  # width
    prediction[..., 3] = torch.exp(prediction[..., 3]) * anchors[:, 1]  # height
    prediction[..., 4] = torch.sigmoid(prediction[..., 4])  # confidence
    prediction[..., 5] = torch.sigmoid(prediction[..., 5])  # class score

    # Przeskaluj x_center i y_center do skali [0, 1]
    prediction[..., 0] = (prediction[..., 0] + grid_x) / grid_w  # x_center w skali [0, 1]
    prediction[..., 1] = (prediction[..., 1] + grid_y) / grid_h  # y_center w skali [0, 1]

    # Konwersja z (x_center, y_center, w, h) na (x_min, y_min, x_max, y_max)
    prediction[..., 0] = prediction[..., 0] - prediction[..., 2] / 2  # x_min
    prediction[..., 1] = prediction[..., 1] - prediction[..., 3] / 2  # y_min
    prediction[..., 2] = prediction[..., 0] + prediction[..., 2]  # x_max
    prediction[..., 3] = prediction[..., 1] + prediction[..., 3]  # y_max

    # Upewnij się, że współrzędne są w granicach [0, 1]
    prediction[..., 0] = torch.clamp(prediction[..., 0], 0, 1)
    prediction[..., 1] = torch.clamp(prediction[..., 1], 0, 1)
    prediction[..., 2] = torch.clamp(prediction[..., 2], 0, 1)
    prediction[..., 3] = torch.clamp(prediction[..., 3], 0, 1)

    # Debugowanie: Wyświetl przykładowe wartości w i h
    print(f"Sample widths (scale {grid_h}x{grid_w}): {prediction[:5, 2]}")
    print(f"Sample heights (scale {grid_h}x{grid_w}): {prediction[:5, 3]}")

    return prediction

def draw_bounding_boxes(img, detections):
    if detections is None or len(detections) == 0:
        return

    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        x_min = int(x_min * img.shape[1])
        y_min = int(y_min * img.shape[0])
        x_max = int(x_max * img.shape[1])
        y_max = int(y_max * img.shape[0])
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

    grid_sizes = [104, 52, 26, 13]  # Rozmiary siatek dla każdej skali

    # Zaktualizowane anchory (usunięto największy anchor)
    anchors = torch.tensor([
        [0.01781505, 0.02762083],
        [0.03658911, 0.05350737],
        [0.06754054, 0.05356116],
        [0.05480609, 0.09285773],
        [0.10250936, 0.10390667],
        [0.08558035, 0.13842838],
        [0.14386341, 0.16825098],
        [0.19815449, 0.29996173]
        # Usunięto [0.4942085, 0.5927835]
    ])

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
                predictions = model(input_img)  # Zwraca listę predykcji z różnych skal
                all_predictions = []
                for scale_idx, pred in enumerate(predictions):
                    B, H, W, num_anchors, _ = pred.shape
                    pred = pred.view(B, -1, 6)[0]  # [num_boxes, 6]
                    confidences = torch.sigmoid(pred[:, 4])
                    print(f"Scale {scale_idx} - Confidence values (top 10): {confidences.topk(10).values}")
                    # Przekształć predykcje dla danej skali
                    pred = convert_to_xyxy(pred, grid_h=grid_sizes[scale_idx], grid_w=grid_sizes[scale_idx], anchors=anchors)
                    all_predictions.append(pred)

                # Połącz predykcje z różnych skal
                all_predictions = torch.cat(all_predictions, dim=0)

                # Wstępna detekcja z wysokim progiem, aby oszacować gęstość
                initial_detections = soft_nms(all_predictions, conf_threshold=0.6, iou_threshold=0.7, sigma=sigma)
                num_detections = len(initial_detections) if len(initial_detections) > 0 else 0

                # Dynamiczne dostosowanie progów
                if num_detections > 50:  # Gęsta scena
                    conf_threshold = 0.3
                    iou_threshold = 0.8
                else:  # Rzadka scena
                    conf_threshold = 0.5
                    iou_threshold = 0.6

                print(f"Using conf_threshold={conf_threshold}, iou_threshold={iou_threshold} for {num_detections} initial detections")
                detections = soft_nms(all_predictions, conf_threshold=conf_threshold, iou_threshold=iou_threshold, sigma=sigma)

                print(f"Before NMS: {len(all_predictions)} boxes, After NMS: {len(detections) if len(detections) > 0 else 0} boxes")

                if save_results:
                    draw_bounding_boxes(orig_img, detections)
                    result_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(result_path, orig_img)
                    print(f"Saved result to {result_path}")

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

if __name__ == "__main__":
    from models.yolo import YOLO
    from utils.non_max_suppression import soft_nms

    model = YOLO(num_classes=1, num_anchors=8)  # Zaktualizowano num_anchors na 8
    detect_images_in_folder("data/images/test/", model, "yolo_model.pth")