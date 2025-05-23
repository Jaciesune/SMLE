import os
import io
import sys
import json
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import argparse
from torchvision.ops import nms

sys.stdout.reconfigure(encoding='utf-8')

def load_model(model_path, device, num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return image, transform(image)

def draw_predictions(image, boxes, labels, scores, threshold):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            box = box.tolist()
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], max(0, box[1] - 15)), f"{label}: {score:.2f}", fill="red", font=font)
    return image

def save_results(image_name, boxes, labels, scores, output_folder, threshold):
    os.makedirs(output_folder, exist_ok=True)

    results = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": image_name,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score)
            })

    json_path = os.path.join(output_folder, f"{image_name}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"[✓] Zapisano wykrycia do {json_path}")
    return json_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Sciezka do obrazu do testowania")
    parser.add_argument("--model_path", required=True, help="Sciezka do wytrenowanego modelu (model_final.pth)")
    parser.add_argument("--output_dir", default="test", help="Folder zapisu wynikow")
    parser.add_argument("--num_classes", type=int, default=2, help="Liczba klas (łacznie z backgroundem)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Minimalny próg ufności dla predykcji")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Uzywane urzadzenie: {device}")

    if not os.path.isfile(args.model_path):
        print(f"[✗] Nie znaleziono modelu: {args.model_path}")
        return

    print(f"[✓] Ladowanie modelu z: {args.model_path}")
    model = load_model(args.model_path, device, num_classes=args.num_classes)

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    image_pil, image_tensor = load_image(args.image_path)

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])

    boxes = prediction[0]['boxes'].cpu()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu()

    # Ograniczenie pokrycia się boxów w obszarze
    keep_indices = nms(boxes, scores, iou_threshold=0.25)
    boxes = boxes[keep_indices].cpu()
    scores = scores[keep_indices].cpu()
    labels = labels[keep_indices].cpu()

    # Wywalanie za dużych
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mean_area = areas.mean()
    min_area = 0.3 * mean_area
    max_area = 3.0 * mean_area

    valid_indices = (areas >= min_area) & (areas <= max_area)
    boxes = boxes[valid_indices].cpu()
    scores = scores[valid_indices].cpu()
    labels = labels[valid_indices].cpu()

    annotated_image = draw_predictions(image_pil.copy(), boxes, labels, scores, args.threshold)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{image_name}_detected.jpg")
    annotated_image.save(save_path)
    print(f"[✓] Zapisano obraz z wykryciami do {save_path}")

    save_results(image_name, boxes, labels, scores, args.output_dir, args.threshold)
    print(f"Detections: {(scores >= args.threshold).sum().item()}")

if __name__ == "__main__":
    main()
