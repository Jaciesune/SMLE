import os
import json
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import argparse
from datetime import datetime


def load_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return image, transform(image)


def draw_predictions(image, boxes, labels, scores, threshold=0.25):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red")
    return image


def save_results(image_name, boxes, labels, scores, output_folder, image_size, threshold=0.25):
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

    print(f"Zapisano wykrycia do {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Ścieżka do obrazu do testowania")
    parser.add_argument("--model_path", default="saved_models/fasterrcnn.pth", help="Ścieżka do wytrenowanego modelu")
    parser.add_argument("--output_dir", default="test", help="Folder zapisu wyników")
    parser.add_argument("--threshold", type=float, default=0.25, help="Próg detekcji")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    image_pil, image_tensor = load_image(args.image_path)

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])

    boxes = prediction[0]['boxes'].cpu()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu()

    annotated_image = draw_predictions(image_pil.copy(), boxes, labels, scores, args.threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, f"{image_name}_pred_{timestamp}.jpg")
    annotated_image.save(save_path)
    print(f"Zapisano obraz z wykryciami do {save_path}")

    save_results(image_name, boxes, labels, scores, args.output_dir, image_pil.size, args.threshold)


if __name__ == "__main__":
    main()
