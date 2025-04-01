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
import numpy as np

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
    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok do wykonania")
    parser.add_argument("--lr", type=float, default=0.0005, help="Początkowa wartość learning rate")
    parser.add_argument("--patience", type=int, default=8, help="Liczba epok bez poprawy dla Early Stopping")
    parser.add_argument("--coco_gt_path", type=str, default="../data/val/annotations/coco.json", help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")
    parser.add_argument("--num_augmentations", type=int, default=8, help="Liczba augmentacji na obraz")
    parser.add_argument("--resume", type=str, default=None, help="Ścieżka do zapisanego checkpointu do wczytania")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Domyślna nazwa modelu z końcówką _checkpoint
    default_model_name = f"train_3_{timestamp}_checkpoint"
    nazwa_modelu = input(f"Podaj nazwę modelu (Enter dla domyślnej: {default_model_name}): ").strip() or default_model_name

    # Upewniamy się, że nazwa modelu kończy się na _checkpoint
    if not nazwa_modelu.endswith('_checkpoint'):
        nazwa_modelu = f"{nazwa_modelu}_checkpoint"

    os.makedirs(f"../logs/train/{nazwa_modelu}", exist_ok=True)
    os.makedirs(f"../logs/val/{nazwa_modelu}", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader = get_data_loaders(
        args.dataset_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        num_augmentations=args.num_augmentations
    )

    # Inicjalizacja list metryk
    straty_treningowe = []
    straty_walidacyjne = []
    liczby_predykcji = []
    liczby_gt = []
    mAPs_bbox = []
    mAPs_seg = []
    start_epoch = 1
    last_epoch = start_epoch - 1  # Domyślna wartość przed rozpoczęciem pętli

    # Ścieżka do pliku _best_checkpoint (do późniejszego usunięcia)
    best_checkpoint_path = None

    # Inicjalizacja modelu, optimizera i scheduler'a
    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    # Wczytywanie checkpointu (jeśli podano --resume)
    if args.resume is not None and os.path.exists(args.resume):
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                straty_treningowe = checkpoint.get('train_losses', [])
                straty_walidacyjne = checkpoint.get('val_losses', [])
                liczby_predykcji = checkpoint.get('num_predictions', [])
                liczby_gt = checkpoint.get('num_gt', [])
                mAPs_bbox = checkpoint.get('mAPs_bbox', [])
                mAPs_seg = checkpoint.get('mAPs_seg', [])
                last_epoch = checkpoint['epoch']
                # Reset learning rate do wartości z args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                print(f"Wczytano checkpoint z {args.resume}. Kontynuuję od epoki {start_epoch} z zresetowanym lr={args.lr}")
            else:
                print(f"Plik {args.resume} nie jest checkpointem. Nie można wczytać jako checkpoint.")
                raise ValueError("Plik nie jest poprawnym checkpointem.")
        except Exception as e:
            print(f"Błąd podczas wczytywania: {e}")
            print("Rozpoczynam trening od nowa")
    else:
        print(f"Wczytano nowy model")

    # Oblicz końcową epokę: start_epoch + liczba dodatkowych epok
    end_epoch = start_epoch + args.epochs - 1  # args.epochs to liczba epok do wykonania

    # Early Stopping
    najlepsza_strata_walidacyjna = float("inf")
    licznik_cierpliwości = 0
    cierpliwość = args.patience

    if start_epoch <= end_epoch:
        for epoch in range(start_epoch, end_epoch + 1):
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

            print(f"Epoka {epoch}/{end_epoch} - Strata treningowa: {strata_treningowa:.4f}, Strata walidacyjna: {strata_walidacyjna:.4f}, Pred: {liczba_predykcji}, GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt} mAP_bbox: {mAP_bbox:.4f}, mAP_seg: {mAP_seg:.4f}")

            # Early Stopping i zapis najlepszego checkpointu
            if strata_walidacyjna < najlepsza_strata_walidacyjna:
                najlepsza_strata_walidacyjna = strata_walidacyjna
                licznik_cierpliwości = 0
                # Zapis checkpointu (dla kontynuacji treningu)
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_losses': straty_treningowe,
                    'val_losses': straty_walidacyjne,
                    'num_predictions': liczby_predykcji,
                    'num_gt': liczby_gt,
                    'mAPs_bbox': mAPs_bbox,
                    'mAPs_seg': mAPs_seg
                }
                best_checkpoint_path = f"../models/{nazwa_modelu}_best_checkpoint.pth"
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Zapisano najlepszy checkpoint: {best_checkpoint_path}")
            else:
                licznik_cierpliwości += 1
                print(f"Brak poprawy przez {licznik_cierpliwości}/{cierpliwość} epok")
                if licznik_cierpliwości >= cierpliwość:
                    print("Early Stopping: Zakończono trening przedwcześnie.")
                    break

            scheduler.step(strata_walidacyjna)
            print(f"Learning rate: {scheduler.get_last_lr()[0]}")
            last_epoch = epoch
    else:
        print(f"start_epoch ({start_epoch}) jest większe lub równe końcowej epoce ({end_epoch + 1}). Trening nie zostanie wykonany.")

    # Zapis końcowego checkpointu
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': last_epoch,
        'train_losses': straty_treningowe,
        'val_losses': straty_walidacyjne,
        'num_predictions': liczby_predykcji,
        'num_gt': liczby_gt,
        'mAPs_bbox': mAPs_bbox,
        'mAPs_seg': mAPs_seg
    }
    nazwa_pliku_checkpointu = f"../models/{nazwa_modelu}_checkpoint.pth"
    torch.save(checkpoint, nazwa_pliku_checkpointu)
    print(f"Checkpoint zapisano jako: {nazwa_pliku_checkpointu}")

    # Usuwanie pliku _best_checkpoint.pth, jeśli istnieje
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
        print(f"Usunięto tymczasowy plik: {best_checkpoint_path}")

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