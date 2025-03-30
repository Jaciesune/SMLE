import torch
import torchvision.models.detection
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from dataset import get_data_loaders
from utils import train_one_epoch, validate_model

# KONFIGURACJA
CONFIDENCE_THRESHOLD = 0.7  # Próg pewności dla detekcji
NMS_THRESHOLD = 0.5  # Ilość propozycji przed NMS
DETECTION_PER_IMAGE = 500  # Maksymalna liczba detekcji na obraz

# Pobranie modelu Mask R-CNN (wersja v2)
def get_model(num_classes, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=num_classes
    )

    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
    model.roi_heads.detections_per_img = DETECTION_PER_IMAGE
    model.to(device)
    print(f"Model działa na: {device}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening Mask R-CNN (v2)")
    parser.add_argument("--dataset_dir", type=str, default="../data", help="Ścieżka do danych")
    parser.add_argument("--num_workers", type=int, default=10, help="Liczba wątków dla DataLoadera")
    parser.add_argument("--batch_size", type=int, default=1, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok")
    parser.add_argument("--lr", type=float, default=0.0005, help="Początkowa wartość learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Liczba epok bez poprawy dla Early Stopping")
    parser.add_argument("--coco_gt_path", type=str, default="../data/val/annotations/coco.json", help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")
    parser.add_argument("--num_augmentations", type=int, default=8, help="Liczba augmentacji na obraz")
    parser.add_argument("--resume", type=str, default=None, help="Ścieżka do zapisanego modelu do wczytania")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nazwa_modelu = input(f"Podaj nazwę modelu (Enter dla domyślnej: mask_rcnn_v2_{timestamp}): ").strip() or f"mask_rcnn_v2_{timestamp}"

    os.makedirs(f"../logs/train/{nazwa_modelu}", exist_ok=True)
    os.makedirs(f"../logs/val/{nazwa_modelu}", exist_ok=True)
    os.makedirs("../logs/models", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader = get_data_loaders(
        args.dataset_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        num_augmentations=args.num_augmentations
    )
    if args.resume is not None and os.path.exists(args.resume):
        try:
            model = torch.load(args.resume, map_location=device, weights_only=False)
            model.to(device)  # Upewnij się, że model jest na odpowiednim urządzeniu
            print(f"Wczytano model z {args.resume}")
        except Exception as e:
            print(f"Błąd podczas wczytywania modelu: {e}")
            model = get_model(num_classes=2, device=device)
            print("Utworzono nowy model z powodu błędu")
    else:
        model = get_model(num_classes=2, device=device)
        print(f"Wczytano nowy model")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    straty_treningowe = []
    straty_walidacyjne = []
    liczby_predykcji = []
    liczby_gt = []
    mAPs_bbox = []
    mAPs_seg = []

    # Early Stopping
    najlepsza_strata_walidacyjna = float("inf")
    licznik_cierpliwości = 0
    cierpliwość = args.patience

    for epoch in range(1, args.epochs + 1):
        strata_treningowa = train_one_epoch(model, train_loader, optimizer, device, epoch)
        strata_walidacyjna, liczba_predykcji, liczba_gt, mAP_bbox, mAP_seg = validate_model(
            model, val_loader, device, epoch, nazwa_modelu, args.coco_gt_path
        )

        straty_treningowe.append(strata_treningowa)
        straty_walidacyjne.append(strata_walidacyjna)
        liczby_predykcji.append(liczba_predykcji)
        liczby_gt.append(liczba_gt)
        mAPs_bbox.append(mAP_bbox)
        mAPs_seg.append(mAP_seg)

        print(f"Epoka {epoch}/{args.epochs} - Strata treningowa: {strata_treningowa:.4f}, Strata walidacyjna: {strata_walidacyjna:.4f}, Pred: {liczba_predykcji}, GT: {liczba_gt}, mAP_bbox: {mAP_bbox:.4f}, mAP_seg: {mAP_seg:.4f}")

        # Early Stopping
        if strata_walidacyjna < najlepsza_strata_walidacyjna:
            najlepsza_strata_walidacyjna = strata_walidacyjna
            licznik_cierpliwości = 0
            torch.save(model, f"../models/{nazwa_modelu}_best.pth")
            print(f"Zapisano najlepszy model: ../models/{nazwa_modelu}_best.pth")
        else:
            licznik_cierpliwości += 1
            print(f"Brak poprawy przez {licznik_cierpliwości}/{cierpliwość} epok")
            if licznik_cierpliwości >= cierpliwość:
                print("Early Stopping: Zakończono trening przedwcześnie.")
                break

        scheduler.step(strata_walidacyjna)
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")

    # Zapis końcowego modelu
    nazwa_pliku_modelu = f"../models/{nazwa_modelu}.pth"
    torch.save(model, nazwa_pliku_modelu)
    print(f"Model zapisano jako: {nazwa_pliku_modelu}")

    # Wykresy
    plt.figure(figsize=(10, 5))
    plt.plot(straty_treningowe, label="Strata treningowa")
    plt.plot(straty_walidacyjne, label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../logs/train/{nazwa_modelu}/loss_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(liczby_predykcji, label="Wykryte obiekty")
    plt.plot(liczby_gt, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../logs/train/{nazwa_modelu}/detections_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(mAPs_bbox, label="mAP (bbox)")
    plt.plot(mAPs_seg, label="mAP (segmentacja)")
    plt.xlabel("Epoka")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../logs/train/{nazwa_modelu}/mAP_plot.png")

    print("Trening zakończony!")