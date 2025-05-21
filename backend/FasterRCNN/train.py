"""
Moduł treningu dla modelu Faster R-CNN

Ten moduł implementuje funkcję trenowania pojedynczej epoki dla modelu 
detekcji obiektów Faster R-CNN. Obsługuje automatyczną kompilację mieszanej
precyzji (mixed precision) dla przyśpieszenia treningu na kompatybilnych GPU,
adaptując się do różnych wersji PyTorch.
"""

#######################
# Importy bibliotek
#######################
import torch                # Framework PyTorch do treningu modeli głębokich sieci neuronowych
import sys                  # Do operacji systemowych
import logging              # Do logowania informacji i błędów

# Import funkcji mixed precision w zależności od wersji PyTorch
if torch.__version__.startswith('2.0'):
    from torch.cuda.amp import autocast, GradScaler   # Dla PyTorch 2.0+
else:
    from torch.amp import autocast, GradScaler        # Dla nowszych wersji PyTorch

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Wymuszenie kodowania UTF-8 dla stdout
sys.stdout.reconfigure(encoding='utf-8')

#######################
# Funkcja treningu
#######################
def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Trenuje model przez jedną epokę.
    
    Przeprowadza pełen cykl treningu modelu na całym zbiorze treningowym,
    obsługując mieszaną precyzję (mixed precision) dla przyspieszenia obliczeń
    na kompatybilnych GPU. Funkcja dostosowuje proces treningu do różnych 
    wersji PyTorch.
    
    Args:
        model (torch.nn.Module): Model Faster R-CNN do trenowania.
        train_loader (torch.utils.data.DataLoader): Loader danych treningowych.
        optimizer (torch.optim.Optimizer): Optymalizator do aktualizacji wag.
        device (torch.device): Urządzenie, na którym ma odbywać się trening.
        epoch (int): Numer aktualnej epoki treningu.
        
    Returns:
        float: Średnia wartość funkcji straty dla całej epoki.
    """
    #######################
    # Przygotowanie treningu
    #######################
    model.train()                    # Przełączenie modelu w tryb treningu
    total_loss = 0.0                 # Inicjalizacja sumy strat
    
    # Inicjalizacja skalera gradientów dla mixed precision (jeśli używamy GPU)
    scaler = GradScaler() if torch.cuda.is_available() else None

    #######################
    # Pętla po partiach danych
    #######################
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Sprawdzenie poprawności danych
        if images is None or targets is None or not images or not targets:
            logger.warning("Pominięto partię z powodu None lub pustych danych.")
            continue

        # Przeniesienie danych na odpowiednie urządzenie
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zerowanie gradientów przed każdą iteracją
        optimizer.zero_grad()

        #######################
        # Mixed precision training
        #######################
        # Wybór odpowiedniego kontekstu w zależności od wersji PyTorch
        if torch.__version__.startswith('2.0'):
            context = autocast() if torch.cuda.is_available() else torch.no_grad()
        else:
            context = autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')

        # Obliczenie strat w kontekście mixed precision
        with context:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        #######################
        # Aktualizacja wag
        #######################
        # Backpropagation z użyciem skalera gradientów dla mixed precision
        if scaler and torch.cuda.is_available():
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standardowa aktualizacja wag dla CPU lub gdy mixed precision jest niedostępna
            losses.backward()
            optimizer.step()

        # Aktualizacja sumy strat
        total_loss += losses.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Logowanie postępu treningu
        logger.info(f"Epoka {epoch}, Partia {batch_idx + 1}/{len(train_loader)}, Strata: {avg_loss:.4f}")

    #######################
    # Podsumowanie epoki
    #######################
    # Obliczenie średniej straty dla całej epoki
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    logger.info(f"Epoka {epoch}, Średnia strata: {avg_loss:.4f}")
    
    return avg_loss