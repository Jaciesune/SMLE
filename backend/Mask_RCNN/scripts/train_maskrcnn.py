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
import shutil
import sys

# KONFIGURACJA
CONFIDENCE_THRESHOLD = 0.7  # Próg pewności dla detekcji
NMS_THRESHOLD = 0.5  # Ilość propozycji przed NMS
DETECTION_PER_IMAGE = 500  # Maksymalna liczba detekcji na obraz

# Pobranie modelu Mask R-CNN (wersja v2)
def get_model(num_classes, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)  # Nie pobieraj wag automatycznie
    # Załaduj wagi ręcznie
    weights_path = "/app/pretrained_weights/maskrcnn_resnet50_fpn_v2.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Plik wag nie istnieje: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
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
    print(f"Model działa na: {device}", flush=True)
    return model

def train_model(args, is_api_call=False):
    """Funkcja treningowa, która może być wywołana z API lub z linii poleceń."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Czy CUDA jest dostępna: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"Nazwa GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Liczba dostępnych GPU: {torch.cuda.device_count()}", flush=True)
    shm_path = "/dev/shm"
    shm_usage = shutil.disk_usage(shm_path)
    print(f"Pamięć współdzielona (/dev/shm):", flush=True)
    print(f"Całkowita: {shm_usage.total / (1024**3):.2f} GB", flush=True)
    print(f"Użyta: {shm_usage.used / (1024**3):.2f} GB", flush=True)
    print(f"Wolna: {shm_usage.free / (1024**3):.2f} GB", flush=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Ustalanie nazwy modelu
    if is_api_call:
        if not args.model_name:
            raise ValueError("Nazwa modelu (--model_name) jest wymagana przy wywołaniu przez API.")
        nazwa_modelu = f"{args.model_name}_{timestamp}_checkpoint"
    else:
        if args.model_name:
            nazwa_modelu = f"{args.model_name}_{timestamp}_checkpoint"
        else:
            default_model_name = f"train_3_{timestamp}_checkpoint"
            try:
                nazwa_modelu_input = input(f"Podaj nazwę modelu (Enter dla domyślnej: {default_model_name}): ").strip()
                nazwa_modelu = nazwa_modelu_input if nazwa_modelu_input else default_model_name
            except EOFError:
                print("Brak interaktywnego wejścia, używam domyślnej nazwy modelu.", flush=True)
                nazwa_modelu = default_model_name

    if not nazwa_modelu.endswith('_checkpoint'):
        nazwa_modelu = f"{nazwa_modelu}_checkpoint"

    # Ścieżki w kontenerze
    os.makedirs(f"/app/logs/train/{nazwa_modelu}", exist_ok=True)
    os.makedirs(f"/app/logs/val/{nazwa_modelu}", exist_ok=True)
    os.makedirs("/app/models", exist_ok=True)

    # Ustalanie ścieżek do folderów danych
    train_dir = args.train_dir  # Używamy train_dir zamiast dataset_dir
    val_dir = "/app/data/val"  # Domyślna ścieżka do danych walidacyjnych

    # Ustalanie ścieżek do plików adnotacji
    coco_train_path = args.coco_train_path if args.coco_train_path else os.path.join(train_dir, "annotations", "coco.json")
    coco_val_path = args.coco_gt_path if args.coco_gt_path else os.path.join(val_dir, "annotations", "coco.json")

    # Sprawdzenie, czy pliki adnotacji istnieją
    if not os.path.exists(coco_train_path):
        raise FileNotFoundError(f"Plik adnotacji treningowych nie istnieje: {coco_train_path}")
    if not os.path.exists(coco_val_path):
        raise FileNotFoundError(f"Plik adnotacji walidacyjnych nie istnieje: {coco_val_path}")

    print("\nWczytywanie danych...", flush=True)
    train_loader, val_loader = get_data_loaders(
        train_dir=train_dir, 
        val_dir=val_dir,      
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        num_augmentations=args.num_augmentations,
        coco_train_path=coco_train_path,  
        coco_val_path=coco_val_path      
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
                print(f"Wczytano checkpoint z {args.resume}. Kontynuuję od epoki {start_epoch} z zresetowanym lr={args.lr}", flush=True)
            else:
                print(f"Plik {args.resume} nie jest checkpointem. Nie można wczytać jako checkpoint.", flush=True)
                raise ValueError("Plik nie jest poprawnym checkpointem.")
        except Exception as e:
            print(f"Błąd podczas wczytywania: {e}", flush=True)
            print("Rozpoczynam trening od nowa", flush=True)
    else:
        print(f"Wczytano nowy model", flush=True)

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
                model, val_loader, device, epoch, nazwa_modelu, coco_val_path
            )

            straty_treningowe.append(strata_treningowa)
            straty_walidacyjne.append(strata_walidacyjna)
            liczby_predykcji.append(liczba_predykcji)
            liczby_gt.append(liczba_gt)
            mAPs_bbox.append(mAP_bbox)
            mAPs_seg.append(mAP_seg)

            print(f"Epoka {epoch}/{end_epoch} - Strata treningowa: {strata_treningowa:.4f}, Strata walidacyjna: {strata_walidacyjna:.4f}, Pred: {liczba_predykcji}, GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt} mAP_bbox: {mAP_bbox:.4f}, mAP_seg: {mAP_seg:.4f}", flush=True)

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
                best_checkpoint_path = f"/app/models/{nazwa_modelu}_best_checkpoint.pth"
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Zapisano najlepszy checkpoint: {best_checkpoint_path}", flush=True)
            else:
                licznik_cierpliwości += 1
                print(f"Brak poprawy przez {licznik_cierpliwości}/{cierpliwość} epok", flush=True)
                if licznik_cierpliwości >= cierpliwość:
                    print("Early Stopping: Zakończono trening przedwcześnie.", flush=True)
                    break

            scheduler.step(strata_walidacyjna)
            print(f"Learning rate: {scheduler.get_last_lr()[0]}", flush=True)
            last_epoch = epoch
    else:
        print(f"start_epoch ({start_epoch}) jest większe lub równe końcowej epoce ({end_epoch + 1}). Trening nie zostanie wykonany.", flush=True)

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
    nazwa_pliku_checkpointu = f"/app/models/{nazwa_modelu}_checkpoint.pth"
    torch.save(checkpoint, nazwa_pliku_checkpointu)
    print(f"Checkpoint zapisano jako: {nazwa_pliku_checkpointu}", flush=True)

    # Usuwanie pliku _best_checkpoint.pth, jeśli istnieje
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
        print(f"Usunięto tymczasowy plik: {best_checkpoint_path}", flush=True)

    # Wykresy
    plt.figure(figsize=(10, 5))
    plt.plot(straty_treningowe, label="Strata treningowa")
    plt.plot(straty_walidacyjne, label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/app/logs/train/{nazwa_modelu}/loss_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(liczby_predykcji, label="Wykryte obiekty")
    plt.plot(liczby_gt, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/app/logs/train/{nazwa_modelu}/detections_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(mAPs_bbox, label="mAP (bbox)")
    plt.plot(mAPs_seg, label="mAP (segmentacja)")
    plt.xlabel("Epoka")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/app/logs/train/{nazwa_modelu}/mAP_plot.png")

    print("Trening zakończony!", flush=True)
    return f"Trening zakończony! Checkpoint zapisany jako: {nazwa_pliku_checkpointu}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening Mask R-CNN (v2)")
    parser.add_argument("--train_dir", type=str, default="/app/train_data", help="Ścieżka do danych treningowych")
    parser.add_argument("--num_workers", type=int, default=1, help="Liczba wątków dla DataLoadera")
    parser.add_argument("--batch_size", type=int, default=1, help="Rozmiar batcha")
    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok do wykonania")
    parser.add_argument("--lr", type=float, default=0.0005, help="Początkowa wartość learning rate")
    parser.add_argument("--patience", type=int, default=8, help="Liczba epok bez poprawy dla Early Stopping")
    parser.add_argument("--coco_train_path", type=str, default=None, help="Ścieżka do pliku COCO z adnotacjami treningowymi")
    parser.add_argument("--coco_gt_path", type=str, default=None, help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")
    parser.add_argument("--num_augmentations", type=int, default=8, help="Liczba augmentacji na obraz")
    parser.add_argument("--resume", type=str, default=None, help="Ścieżka do zapisanego checkpointu do wczytania")
    parser.add_argument("--model_name", type=str, default=None, help="Nazwa modelu (wymagane przy wywołaniu przez API)")

    args = parser.parse_args()
    train_model(args, is_api_call=False)