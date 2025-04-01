from ultralytics import YOLO
import argparse
import os
from datetime import datetime
from glob import glob
import cv2

def list_models():
    model_paths = glob("runs/*/weights/best.pt")
    model_paths = sorted(model_paths)
    if not model_paths:
        print("Nie znaleziono żadnych modeli w runs/*/weights/best.pt")
        exit(1)

    print("Dostępne modele YOLOv8:")
    for idx, path in enumerate(model_paths):
        print(f"  [{idx}] {path}")
    return model_paths

def main():
    parser = argparse.ArgumentParser(description="Testowanie modelu YOLOv8 na obrazach testowych")
    parser.add_argument("--name", type=str, help="Nazwa folderu wyników (Enter dla domyślnej)")

    args = parser.parse_args()

    # Lista modeli do wyboru
    model_paths = list_models()
    while True:
        try:
            choice = int(input("\nWybierz numer modelu do przetestowania: "))
            model_path = model_paths[choice]
            break
        except (ValueError, IndexError):
            print("Niepoprawny wybór. Wprowadź poprawny numer z listy.")

    # Nazwa folderu wyników
    args.name = args.name or input("Podaj nazwę folderu wyników (Enter dla domyślnej): ").strip() or f"predict_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_dir = os.path.join("runs", "test_results", args.name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nWybrany model: {model_path}")
    model = YOLO(model_path)

    print(f"Rozpoczynam predykcję na zbiorze testowym...")

    results = model.predict(
        source="dataset/images/test",
        save=True,
        save_txt=False,
        show_labels=False,
        show_conf=False,
        project=save_dir,
        name="",
        exist_ok=True,
        verbose=False
    )

    print(f"\nTworzenie plików tekstowych z liczbą wykrytych rur...")

    for result in results:
        path = result.path  # pełna ścieżka do oryginalnego obrazu
        boxes = result.boxes
        count = len(boxes)

        base_name = os.path.splitext(os.path.basename(path))[0]
        txt_path = os.path.join(save_dir, f"{base_name}.txt")

        with open(txt_path, "w") as f:
            f.write(f"{count}\n")

    print(f"\nPredykcja zakończona. Wyniki i pliki .txt zapisano w: {save_dir}")

if __name__ == "__main__":
    main()
