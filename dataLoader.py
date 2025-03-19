import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import multiprocessing

# Domyślna liczba wątków
DEFAULT_NUM_WORKERS = min(4, multiprocessing.cpu_count() - 2)  # Optymalizacja dla CPU

# Argumenty wiersza poleceń
parser = argparse.ArgumentParser(description="Konfiguracja DataLoadera")
parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Liczba wątków do przetwarzania danych")
args, _ = parser.parse_known_args()  # Pobieranie argumentów bez błędów w innych skryptach

# Pobieranie liczby wątków z argumentu
num_workers = max(0, args.num_workers)  # Zapewnienie, że nie ma wartości ujemnych

# Ścieżki do zbiorów treningowych i testowych
train_images = "dataset/train/images"
train_annotations = "dataset/train/annotations.json"
test_images = "dataset/test/images"
test_annotations = "dataset/test/annotations.json"

# Transformacje obrazu (konwersja do tensora)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Funkcja obsługi błędów w ładowaniu anotacji
def load_dataset(images_path, annotations_path, transform):
    try:
        dataset = CocoDetection(root=images_path, annFile=annotations_path, transform=transform)
        return dataset
    except Exception as e:
        print(f"Błąd wczytywania zbioru {annotations_path}: {e}")
        return None

# Wczytanie zbioru treningowego i testowego
train_dataset = load_dataset(train_images, train_annotations, transform)
test_dataset = load_dataset(test_images, test_annotations, transform)

# Funkcja collate_fn do grupowania batchy
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# Pobieranie DataLoaderów
def get_data_loaders():
    if train_dataset is None or test_dataset is None:
        print("Błąd: Jeden z datasetów nie został poprawnie załadowany!")
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    print(f"DataLoader gotowy! Trening: {len(train_dataset)} obrazów, Test: {len(test_dataset)} obrazów")
    print(f"Używana liczba wątków: {num_workers}")
    
    return train_loader, test_loader

# Uruchamianie testowe
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()