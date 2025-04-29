# train.py
import torch
import sys
import logging

# Wybór odpowiedniego API dla autocast w zależności od wersji PyTorch
if torch.__version__.startswith('2.0'):
    from torch.cuda.amp import autocast, GradScaler
else:
    from torch.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding='utf-8')

def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    scaler = GradScaler() if torch.cuda.is_available() else None

    for batch_idx, (images, targets) in enumerate(train_loader):
        if images is None or targets is None:
            logger.warning("Pominięto partię z powodu None w obrazach lub targetach.")
            continue

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Normalizacja obrazów
        images = [(image.float() / 255.0) for image in images]

        optimizer.zero_grad()

        # Użycie autocast w zależności od wersji PyTorch
        if torch.__version__.startswith('2.0'):
            context = autocast() if torch.cuda.is_available() else torch.no_grad()
        else:
            context = autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')

        with context:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if scaler and torch.cuda.is_available():
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        total_loss += losses.item()
        avg_loss = total_loss / (batch_idx + 1)  # Średnia strata dla bieżącej liczby partii
        logger.info(f"Epoka {epoch}, Partia {batch_idx + 1}/{len(train_loader)}, Strata: {avg_loss:.4f}")

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    logger.info(f"Epoka {epoch}, Średnia strata: {avg_loss:.4f}")
    return avg_loss