import os
from pathlib import Path
import subprocess
import shutil

class DetectionAPI:
    def __init__(self):
        # Ścieżka bazowa do folderu backend
        self.base_path = Path(__file__).resolve().parent.parent  # backend/
        self.detectes_path = self.base_path / "Mask_RCNN" / "data" / "detectes"
        self.detectes_path.mkdir(parents=True, exist_ok=True)

        # Definicja algorytmów i ich folderów z modelami
        self.algorithms = {
            "Mask R-CNN": self.base_path / "Mask_RCNN" / "models",
            "Faster R-CNN": self.base_path / "Faster_RCNN" / "models",
            "YOLO": self.base_path / "YOLO" / "models"
        }

    def get_algorithms(self):
        """Zwraca listę dostępnych algorytmów."""
        return list(self.algorithms.keys())

    def get_model_versions(self, algorithm):
        """Zwraca listę plików modeli z końcówką *checkpoint.pth dla wybranego algorytmu."""
        if algorithm not in self.algorithms:
            return []

        model_path = self.algorithms[algorithm]
        if not model_path.exists():
            return []

        # Pobierz tylko pliki z końcówką *checkpoint.pth w folderze models
        model_versions = [file.name for file in model_path.iterdir() if file.is_file() and file.name.endswith('_checkpoint.pth')]
        return sorted(model_versions)

    def get_model_path(self, algorithm, version):
        """Zwraca pełną ścieżkę do wybranego modelu."""
        if algorithm not in self.algorithms:
            return None

        model_path = self.algorithms[algorithm] / version
        if model_path.exists() and model_path.is_file() and model_path.name.endswith('_checkpoint.pth'):
            return str(model_path)
        return None

    def analyze_with_model(self, image_path, algorithm, version):
        """Przeprowadza detekcję na obrazie przy użyciu wybranego modelu. Zwraca ścieżkę do wyniku i liczbę wykrytych obiektów."""
        model_path = self.get_model_path(algorithm, version)
        if not model_path:
            return f"Błąd: Model {version} dla algorytmu {algorithm} nie istnieje lub nie kończy się na _checkpoint.pth.", None

        if not os.path.exists(image_path):
            return f"Błąd: Obraz {image_path} nie istnieje.", None

        # Kopiujemy obraz do folderu data/test/images, aby detect.py mógł go przetworzyć
        test_images_path = self.base_path / "Mask_RCNN" / "data" / "test" / "images"
        test_images_path.mkdir(parents=True, exist_ok=True)
        image_name = os.path.basename(image_path)
        temp_image_path = test_images_path / image_name
        shutil.copy(image_path, temp_image_path)

        # Ścieżka do skryptu detect.py w kontenerze
        detect_script_path = self.base_path / "Mask_RCNN" / "scripts" / "detect.py"

        # Uruchamiamy detect.py w tym samym środowisku
        try:
            result = subprocess.run([
                "python",
                str(detect_script_path),
                str(temp_image_path),
                str(model_path)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return f"Błąd podczas detekcji: {result.stderr}", None

            # Ścieżka do wyniku detekcji
            result_image_name = os.path.splitext(image_name)[0] + "_detected.jpg"
            result_path = self.detectes_path / result_image_name

            # Ścieżka do pliku z liczbą wykrytych obiektów
            result_count_path = self.detectes_path / (os.path.splitext(image_name)[0] + "_detected.txt")

            if result_path.exists() and result_count_path.exists():
                # Odczyt liczby wykrytych obiektów
                with open(result_count_path, 'r') as f:
                    detections_count = int(f.read().strip())
                return str(result_path), detections_count
            else:
                return f"Błąd: Wynik detekcji nie został zapisany w {result_path} lub brak pliku z liczbą wykrytych obiektów.", None

        except subprocess.CalledProcessError as e:
            return f"Błąd podczas uruchamiania skryptu detect.py: {e}", None
        finally:
            # Usuwamy tymczasowy obraz
            if temp_image_path.exists():
                temp_image_path.unlink()