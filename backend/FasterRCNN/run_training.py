import torch
import torch.optim as optim
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader import get_data_loaders
from model import get_model
from train import train_one_epoch
from val_utils import validate_model

from config import CONFIDENCE_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN z ResNet50")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, help="Wielkość batcha")
    parser.add_argument("--epochs", type=int, help="Liczba epok")
    parser.add_argument("--model_name", type=str, help="Nazwa modelu")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Interaktywny input jeśli brak argumentów
    model_name = args.model_name or input(f"Podaj nazwę modelu (Enter dla domyślnej): ").strip() or f"faster_rcnn_{timestamp}"
    batch_size = args.batch_size or int(input("Podaj batch size (domyślnie 2): ") or 2)
    epochs = args.epochs or int(input("Podaj liczbę epok (domyślnie 40): ") or 40)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUżywane urządzenie: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    os.makedirs(f"train/{model_name}", exist_ok=True)
    os.makedirs(f"val/{model_name}", exist_ok=True)
    os.makedirs(f"saved_models/{model_name}", exist_ok=True)
    os.makedirs(f"test/{model_name}", exist_ok=True)

    print("\nWczytywanie danych...")
    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size, num_workers=args.num_workers)

    model = get_model(num_classes=2, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_val_loss = float("inf")
    best_epoch = 0
    train_losses, val_losses, pred_counts, gt_counts = [], [], [], []
    map_5095_list, map_50_list, precision_list, recall_list = [], [], [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, pred_count, gt_count, map_5095, map_50, precision, recall = validate_model(model, val_loader, device, epoch, model_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        map_5095_list.append(map_5095)
        map_50_list.append(map_50)
        precision_list.append(precision)
        recall_list.append(recall)

        print(f"Epoka {epoch}/{epochs} - Strata treningowa: {train_loss:.4f}, Strata walidacyjna: {val_loss:.4f}")
        print(f"                - Detekcje: {pred_count} | GT: {gt_count}")
        print(f"                - mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_5095:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"saved_models/{model_name}/model_epoch_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"saved_models/{model_name}/model_best_epoch_{epoch}.pth")

    # Zapis modelu końcowego
    torch.save(model.state_dict(), f"saved_models/{model_name}/model_final.pth")
    print(f"\nModel końcowy zapisano jako: saved_models/{model_name}/model_final.pth")
    print(f"Najlepszy model pochodzi z epoki {best_epoch} (val_loss = {best_val_loss:.4f})")

    # Wykres strat
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Strata treningowa")
    plt.plot(val_losses, label="Strata walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.title("Strata w czasie treningu")
    plt.savefig(f"test/{model_name}/loss_plot.png")
    plt.close()

    # Wykres liczby detekcji
    plt.figure(figsize=(10, 5))
    plt.plot(pred_counts, label="Wykryte obiekty")
    plt.plot(gt_counts, label="Obiekty GT")
    plt.xlabel("Epoka")
    plt.ylabel("Liczba obiektów")
    plt.legend()
    plt.grid(True)
    plt.title("Porównanie predykcji i GT")
    plt.savefig(f"test/{model_name}/detections_plot.png")
    plt.close()

    # Wykres mAP
    plt.figure(figsize=(10, 5))
    plt.plot(map_50_list, label="mAP@0.5")
    plt.plot(map_5095_list, label="mAP@0.5:0.95")
    plt.xlabel("Epoka")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"test/{model_name}/map_plot.png")
    plt.close()

    print("Wykresy zapisane w folderze test.")


if __name__ == "__main__":
    main()