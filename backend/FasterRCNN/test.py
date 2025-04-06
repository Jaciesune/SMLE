import os
import json
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import argparse
from datetime import datetime


def load_model(model_path, device, num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return image, transform(image)


def draw_predictions(image, boxes, labels, scores, threshold=0.25):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            box = box.tolist()
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill="red", font=font)
    return image


def save_results(image_name, boxes, labels, scores, output_folder, threshold=0.25):
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
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"[✓] Zapisano wykrycia do {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Ścieżka do obrazu do testowania")
    parser.add_argument("--model_path", default="saved_models/fasterrcnn.pth", help="Ścieżka do wytrenowanego modelu")
    parser.add_argument("--output_dir", default="test", help="Folder zapisu wyników")
    parser.add_argument("--threshold", type=float, default=0.25, help="Próg detekcji")
    parser.add_argument("--num_classes", type=int, default=2, help="Liczba klas (łącznie z backgroundem)")
    args = parser.parse_args()

    # Sprawdzenie GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Używane urządzenie: {device}")

    # Wczytanie modelu
    print(f"[✓] Ładowanie modelu z: {args.model_path}")
    model = load_model(args.model_path, device, num_classes=args.num_classes)

    # Wczytanie obrazu
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    image_pil, image_tensor = load_image(args.image_path)

    # Predykcja
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])

    boxes = prediction[0]['boxes'].cpu()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu()

    # Rysowanie i zapis
    annotated_image = draw_predictions(image_pil.copy(), boxes, labels, scores, args.threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, f"{image_name}_detected.jpg")
    os.makedirs(args.output_dir, exist_ok=True)
    annotated_image.save(save_path)
    print(f"[✓] Zapisano obraz z wykryciami do {save_path}")

    # JSON
    save_results(image_name, boxes, labels, scores, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
