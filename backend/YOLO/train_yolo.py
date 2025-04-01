from ultralytics import YOLO
import argparse
import os
import torch
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Trening YOLOv8 dla detekcji rur")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Ścieżka do pliku data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrenowany model bazowy (np. yolov8n.pt)")
    parser.add_argument("--epochs", type=int, help="Liczba epok")
    parser.add_argument("--imgsz", type=int, default=1024, help="Rozmiar wejściowego obrazu")
    parser.add_argument("--batch", type=int, help="Wielkość batcha")
    parser.add_argument("--name", type=str, help="Nazwa eksperymentu")

    args = parser.parse_args()

    # Interaktywny input jeśli brak argumentów
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.name = args.name or input("Podaj nazwę eksperymentu (Enter dla domyślnej): ").strip() or f"yolo_rury_{timestamp}"
    args.epochs = args.epochs or int(input("Podaj liczbę epok (Enter dla domyślnej 40): ") or 40)
    args.batch = args.batch or int(input("Podaj batch size (Enter dla domyślnej 2): ") or 2)

    print(f"\nRozpoczynam trening YOLOv8: {args.name}")
    print("Używane urządzenie:", "CUDA" if torch.cuda.is_available() else "CPU")
    print(f"Model bazowy: {args.model}")
    print(f"Dane: {args.data}")
    print(f"Epoki: {args.epochs}, Batch: {args.batch}, Rozmiar obrazu: {args.imgsz}\n")

    # Wczytaj model
    model = YOLO(args.model)

    # Trening
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device="cuda",
        project="runs"
    )

    print(f"Trening zakończony. Wyniki znajdziesz w: runs/detect/{args.name}")


if __name__ == "__main__":
    main()
