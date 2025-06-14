"""
Skrypt treningu modelu Mask R-CNN dla segmentacji instancji

Ten skrypt implementuje pełny proces treningu modelu Mask R-CNN opartego na ResNet-50
z Feature Pyramid Network. Umożliwia wznowienie treningu z checkpointu, automatyczną
estymację parametrów oraz monitorowanie metryk wydajności w czasie rzeczywistym.
Zawiera również mechanizmy early stopping, redukcji learning rate oraz
wizualizację postępów treningu.
"""

#######################
# Importy bibliotek
#######################
import torch                                # Framework PyTorch
import torchvision.models.detection         # Modele detekcji obiektów
import torch.optim as optim                 # Optymalizatory
import torch.optim.lr_scheduler as lr_scheduler  # Harmonogramy zmian learning rate
import matplotlib.pyplot as plt             # Do wizualizacji wyników
import os                                   # Operacje na systemie plików
import argparse                             # Do parsowania argumentów
from datetime import datetime               # Do obsługi dat i czasu
from dataset import get_data_loaders        # Funkcja wczytywania danych
from utils import train_one_epoch, validate_model  # Pomocnicze funkcje treningu
import numpy as np                          # Operacje numeryczne
import shutil                               # Operacje na plikach
import sys                                  # Operacje systemowe
import signal                               # Obsługa sygnałów systemowych
from tqdm import tqdm                       # Paski postępu

#######################
# Konfiguracja parametrów
#######################
# Parametry modelu
CONFIDENCE_THRESHOLD = 0.5  # Obniżony próg pewności
NMS_THRESHOLD = 1600        # Liczba propozycji przed i po NMS
DETECTION_PER_IMAGE = 800   # Maksymalna liczba detekcji na obraz
NUM_CLASSES = 2             # Liczba klas (tło + 1 klasa obiektów)

# Ścieżki do katalogów
BASE_DIR = "/app/backend/Mask_RCNN"
PRETRAINED_WEIGHTS_DIR = os.path.join(BASE_DIR, "pretrained_weights")
PRETRAINED_WEIGHTS_PATH = os.path.join(PRETRAINED_WEIGHTS_DIR, "maskrcnn_resnet50_fpn_v2.pth")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_TRAIN_DIR = os.path.join(BASE_DIR, "logs/train")
LOGS_VAL_DIR = os.path.join(BASE_DIR, "logs/val")
DEFAULT_TRAIN_DIR = "/app/backend/data/train"
DEFAULT_VAL_DIR = "/app/backend/data/val"

# Domyślne parametry treningu
DEFAULT_EPOCHS = 20
DEFAULT_LR = 0.0005
DEFAULT_PATIENCE = 9
DEFAULT_NUM_AUGMENTATIONS = 2

# Parametry optymalizatora
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 4
MIN_LR = 5e-7

#######################
# Funkcje pomocnicze
#######################
def parse_args():
    """
    Parsuje argumenty linii poleceń.
    
    Returns:
        argparse.Namespace: Sparsowane argumenty.
    """
    parser = argparse.ArgumentParser(description="Trening Mask R-CNN (v2)")
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR, help="Ścieżka do danych treningowych")
    parser.add_argument("--val_dir", type=str, default=DEFAULT_VAL_DIR, help="Ścieżka do danych walidacyjnych")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Liczba epok do wykonania")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Początkowa wartość learning rate")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Liczba epok bez poprawy dla Early Stopping")
    parser.add_argument("--coco_train_path", type=str, default=None, help="Ścieżka do pliku COCO z adnotacjami treningowymi")
    parser.add_argument("--coco_gt_path", type=str, default=None, help="Ścieżka do pliku COCO z adnotacjami walidacyjnymi")
    parser.add_argument("--num_augmentations", type=int, default=DEFAULT_NUM_AUGMENTATIONS, help="Liczba augmentacji na obraz")
    parser.add_argument("--resume", type=str, default=None, help="Nazwa modelu do wczytania (bez ścieżki)")
    parser.add_argument("--model_name", type=str, default=None, help="Nazwa modelu (wymagane przy wywołaniu przez API)")
    return parser.parse_args()

def download_pretrained_weights():
    """
    Pobiera i zapisuje wagi pretrenowanego modelu Mask R-CNN.
    """
    print(f"Pobieranie pretrained weights do {PRETRAINED_WEIGHTS_PATH}...", flush=True)
    try:
        os.makedirs(PRETRAINED_WEIGHTS_DIR, exist_ok=True)
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
        torch.save(model.state_dict(), PRETRAINED_WEIGHTS_PATH)
        print(f"Pretrained weights zapisano jako: {PRETRAINED_WEIGHTS_PATH}", flush=True)
    except Exception as e:
        print(f"Błąd podczas pobierania pretrained weights: {str(e)}", flush=True)
        raise

def get_model(num_classes, device):
    """
    Inicjalizuje model Mask R-CNN z pretrenowanymi wagami i dostosowuje go do zadania.
    
    Args:
        num_classes (int): Liczba klas (łącznie z tłem).
        device (torch.device): Urządzenie, na którym ma działać model.
        
    Returns:
        torch.nn.Module: Skonfigurowany model Mask R-CNN.
    """
    try:
        if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
            print(f"Pretrained weights nie istnieją w {PRETRAINED_WEIGHTS_PATH}. Pobieranie...", flush=True)
            download_pretrained_weights()
        else:
            print(f"Używam istniejących pretrained weights z {PRETRAINED_WEIGHTS_PATH}", flush=True)

        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)
        state_dict = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
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
    except Exception as e:
        print(f"Błąd podczas inicjalizacji modelu: {str(e)}", flush=True)
        raise

def signal_handler(sig, frame):
    """
    Obsługuje sygnały systemowe, takie jak SIGTERM, aby umożliwić czyszczenie VRAM.
    
    Args:
        sig: Otrzymany sygnał.
        frame: Ramka stosu w momencie otrzymania sygnału.
    """
    print(f"Otrzymano sygnał {sig}, czyszczenie zasobów...", flush=True)
    sys.exit(0)  # Wyjdź z programu, co wywoła sekcję finally w train_model

def train_model(args, is_api_call=False):
    """
    Główna funkcja treningu modelu Mask R-CNN.
    
    Implementuje pełny cykl treningu, w tym:
    - Inicjalizację/wczytywanie modelu
    - Ładowanie danych
    - Trenowanie modelu przez określoną liczbę epok
    - Walidację modelu
    - Zapisywanie checkpointów
    - Wizualizację wyników
    
    Args:
        args (argparse.Namespace): Sparsowane argumenty.
        is_api_call (bool): Czy funkcja jest wywoływana przez API.
        
    Returns:
        str: Komunikat o zakończeniu treningu.
    """
    # Zarejestruj obsługę sygnału SIGTERM
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        #######################
        # Konfiguracja urządzenia i diagnostyka
        #######################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("Włączono torch.backends.cudnn.benchmark dla optymalizacji GPU", flush=True)
            print(f"Nazwa GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Liczba dostępnych GPU: {torch.cuda.device_count()}", flush=True)
        
        shm_path = "/dev/shm"
        shm_usage = shutil.disk_usage(shm_path)
        print(f"Pamięć współdzielona (/dev/shm):", flush=True)
        print(f"Całkowita: {shm_usage.total / (1024**3):.2f} GB", flush=True)
        print(f"Użyta: {shm_usage.used / (1024**3):.2f} GB", flush=True)
        print(f"Wolna: {shm_usage.free / (1024**3):.2f} GB", flush=True)

        #######################
        # Ustalenie nazwy modelu
        #######################
        if is_api_call:
            if not args.model_name:
                raise ValueError("Nazwa modelu (--model_name) jest wymagana przy wywołaniu przez API.")
            base_model_name = args.model_name
        else:
            if args.model_name:
                base_model_name = args.model_name
            else:
                default_model_name = "train_3"
                try:
                    nazwa_modelu_input = input(f"Podaj nazwę modelu (Enter dla domyślnej: {default_model_name}): ").strip()
                    base_model_name = nazwa_modelu_input if nazwa_modelu_input else default_model_name
                except EOFError:
                    print("Brak interaktywnego wejścia, używam domyślnej nazwy modelu.", flush=True)
                    base_model_name = default_model_name

        #######################
        # Tworzenie katalogów
        #######################
        os.makedirs(os.path.join(LOGS_TRAIN_DIR, base_model_name), exist_ok=True)
        os.makedirs(os.path.join(LOGS_VAL_DIR, base_model_name), exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        #######################
        # Konfiguracja ścieżek danych
        #######################
        train_dir = args.train_dir
        val_dir = args.val_dir

        coco_train_path = args.coco_train_path if args.coco_train_path else os.path.join(train_dir, "annotations", "instances_train.json")
        coco_val_path = args.coco_gt_path if args.coco_gt_path else os.path.join(val_dir, "annotations", "instances_val.json")

        if not os.path.exists(coco_train_path):
            raise FileNotFoundError(f"Plik adnotacji treningowych nie istnieje: {coco_train_path}")
        if not os.path.exists(coco_val_path):
            raise FileNotFoundError(f"Plik adnotacji walidacyjnych nie istnieje: {coco_val_path}")

        #######################
        # Wczytanie danych
        #######################
        print("\nWczytywanie danych...", flush=True)
        train_loader, val_loader = get_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            num_augmentations=args.num_augmentations,
            coco_train_path=coco_train_path,
            coco_val_path=coco_val_path
        )

        #######################
        # Inicjalizacja struktur do śledzenia postępu
        #######################
        straty_treningowe = []
        straty_walidacyjne = []
        liczby_predykcji = []
        liczby_gt = []
        mAPs_bbox = []
        mAPs_seg = []
        start_epoch = 1
        last_epoch = start_epoch - 1

        best_checkpoint_path = os.path.join(MODELS_DIR, f"{base_model_name}_bestepoch_checkpoint.pth")

        #######################
        # Inicjalizacja modelu, optymalizatora i scheduler'a
        #######################
        model = get_model(num_classes=NUM_CLASSES, device=device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
        )

        #######################
        # Wczytanie checkpointu (jeśli podano)
        #######################
        resume_path = None
        if args.resume:
            resume_path = os.path.join(MODELS_DIR, args.resume)
            print(f"Argument --resume: {args.resume}, pełna ścieżka: {resume_path}", flush=True)
        else:
            print("Brak argumentu --resume, wczytuję nowy model", flush=True)

        if resume_path and os.path.exists(resume_path):
            try:
                checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
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
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
                    print(f"Wczytano checkpoint z {resume_path}. Kontynuuję od epoki {start_epoch} z zresetowanym lr={args.lr}", flush=True)
                else:
                    raise ValueError("Plik nie jest poprawnym checkpointem.")
            except Exception as e:
                print(f"Błąd podczas wczytywania checkpointu: {str(e)}", flush=True)
                print("Rozpoczynam trening od nowa", flush=True)
        else:
            print(f"Wczytano nowy model", flush=True)

        #######################
        # Obliczenie zakresu epok
        #######################
        end_epoch = start_epoch + args.epochs - 1

        #######################
        # Inicjalizacja parametrów early stopping
        #######################
        najlepsza_strata_walidacyjna = float("inf")
        licznik_cierpliwości = 0
        cierpliwość = args.patience

        #######################
        # Główna pętla treningowa
        #######################
        if start_epoch <= end_epoch:
            with tqdm(total=end_epoch - start_epoch + 1, desc="Trening", unit="epoka", file=sys.stdout) as pbar:
                for epoch in range(start_epoch, end_epoch + 1):
                    try:
                        # Trening na jednej epoce
                        strata_treningowa = train_one_epoch(
                            model, train_loader, optimizer, device, epoch
                        )
                        
                        # Walidacja modelu
                        strata_walidacyjna, liczba_predykcji, liczba_gt, mAP_bbox, mAP_seg = validate_model(
                            model, val_loader, device, epoch, base_model_name, coco_val_path
                        )

                        # Zapisanie metryk
                        straty_treningowe.append(strata_treningowa)
                        straty_walidacyjne.append(strata_walidacyjna)
                        liczby_predykcji.append(liczba_predykcji)
                        liczby_gt.append(liczba_gt)
                        mAPs_bbox.append(mAP_bbox)
                        mAPs_seg.append(mAP_seg)

                        # Aktualizacja informacji o postępie
                        log_message = (
                            f"Epoka {epoch}/{end_epoch} - Strata treningowa: {strata_treningowa:.4f}, "
                            f"Strata walidacyjna: {strata_walidacyjna:.4f}, Pred: {liczba_predykcji}, "
                            f"GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt:.2f}, "
                            f"mAP_bbox: {mAP_bbox:.4f}, mAP_seg: {mAP_seg:.4f}"
                        )
                        pbar.set_description(log_message)
                        pbar.update(1)

                        # Sprawdzenie czy model się poprawił
                        if strata_walidacyjna < najlepsza_strata_walidacyjna:
                            najlepsza_strata_walidacyjna = strata_walidacyjna
                            licznik_cierpliwości = 0
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
                            torch.save(checkpoint, best_checkpoint_path)
                            print(f"Zapisano najlepszy checkpoint: {best_checkpoint_path}", flush=True)
                        else:
                            licznik_cierpliwości += 1
                            print(f"Brak poprawy przez {licznik_cierpliwości}/{cierpliwość} epok", flush=True)
                            if licznik_cierpliwości >= cierpliwość:
                                print("Early Stopping: Zakończono trening przedwcześnie.", flush=True)
                                break

                        # Aktualizacja learning rate
                        scheduler.step(strata_walidacyjna)
                        print(f"Learning rate: {scheduler.get_last_lr()[0]}", flush=True)
                        last_epoch = epoch
                    except Exception as e:
                        print(f"Błąd podczas epoki {epoch}: {str(e)}", flush=True)
                        raise
        else:
            print(f"start_epoch ({start_epoch}) jest większe lub równe końcowej epoce ({end_epoch + 1}). Trening nie zostanie wykonany.", flush=True)

        #######################
        # Zapisanie finalnego checkpointu
        #######################
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
        nazwa_pliku_checkpointu = os.path.join(MODELS_DIR, f"{base_model_name}_checkpoint.pth")
        torch.save(checkpoint, nazwa_pliku_checkpointu)
        print(f"Checkpoint zapisano jako: {nazwa_pliku_checkpointu}", flush=True)

        # Usunięcie tymczasowego najlepszego checkpointu
        if os.path.exists(best_checkpoint_path):
            os.remove(best_checkpoint_path)
            print(f"Usunięto tymczasowy plik: {best_checkpoint_path}", flush=True)

        #######################
        # Generowanie wykresów
        #######################
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(straty_treningowe, label="Strata treningowa")
            plt.plot(straty_walidacyjne, label="Strata walidacyjna")
            plt.xlabel("Epoka")
            plt.ylabel("Strata")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(LOGS_TRAIN_DIR, base_model_name, "loss_plot.png"))

            plt.figure(figsize=(10, 5))
            plt.plot(liczby_predykcji, label="Wykryte obiekty")
            plt.plot(liczby_gt, label="Obiekty GT")
            plt.xlabel("Epoka")
            plt.ylabel("Liczba obiektów")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(LOGS_TRAIN_DIR, base_model_name, "detections_plot.png"))

            plt.figure(figsize=(10, 5))
            plt.plot(mAPs_bbox, label="mAP (bbox)")
            plt.plot(mAPs_seg, label="mAP (segmentacja)")
            plt.xlabel("Epoka")
            plt.ylabel("mAP")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(LOGS_TRAIN_DIR, base_model_name, "mAP_plot.png"))
        except Exception as e:
            print(f"Błąd podczas generowania wykresów: {str(e)}", flush=True)
            raise

        print("Trening zakończony!", flush=True)
        return "Trening zakończony sukcesem!"
    except Exception as e:
        error_message = f"Trening zakończył się błędem: {str(e)}"
        print(error_message, flush=True)
        raise Exception(error_message)
    finally:
        # Czyszczenie pamięci GPU i zasobów
        if torch.cuda.is_available():
            try:
                # Usunięcie referencji do modelu i optymalizatora
                del model
                del optimizer
                del scheduler
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Synchronizacja GPU dla pewności
                print("Zwolniono pamięć VRAM i usunięto obiekty modelu", flush=True)
            except Exception as e:
                print(f"Błąd podczas czyszczenia pamięci VRAM: {str(e)}", flush=True)
        
        # Zwolnienie zasobów DataLoader
        try:
            if 'train_loader' in locals():
                if hasattr(train_loader.dataset, '_shut_down_workers'):
                    train_loader.dataset._shut_down_workers()
                    print("Zamknięto procesy DataLoader dla train_loader", flush=True)
            if 'val_loader' in locals():
                if hasattr(val_loader.dataset, '_shut_down_workers'):
                    val_loader.dataset._shut_down_workers()
                    print("Zamknięto procesy DataLoader dla val_loader", flush=True)
        except Exception as e:
            print(f"Błąd podczas zamykania procesów DataLoader: {str(e)}", flush=True)
        
        # Dodatkowe czyszczenie pamięci systemowej
        import gc
        gc.collect()
        print("Wykonano garbage collection", flush=True)

#######################
# Punkt wejścia programu
#######################
if __name__ == "__main__":
    args = parse_args()
    try:
        result = train_model(args, is_api_call=False)
        print(result, flush=True)
    except Exception as e:
        print(f"Błąd końcowy: {str(e)}", flush=True)
        raise