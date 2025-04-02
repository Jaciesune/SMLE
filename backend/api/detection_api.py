import os
from pathlib import Path
import subprocess
import shutil
import re

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
        """Zwraca listę plików modeli z końcówką *_checkpoint.pth dla wybranego algorytmu."""
        if algorithm not in self.algorithms:
            return []

        model_path = self.algorithms[algorithm]
        if not model_path.exists():
            return []

        # Pobierz tylko pliki z końcówką *_checkpoint.pth w folderze models
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

    def run_maskrcnn_script(self, script_name, *args):
        """Uruchamia skrypt Mask R-CNN w kontenerze maskrcnn."""
        try:
            # Przygotowanie polecenia docker run
            command = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{self.base_path}/Mask_RCNN:/app",
                "smle-maskrcnn",
                "python", f"scripts/{script_name}", *args
            ]
            print(f"Uruchamiam polecenie: {' '.join(command)}")  # Logowanie dla debugowania
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Błąd podczas uruchamiania skryptu {script_name}: {result.stderr}"
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Błąd podczas uruchamiania kontenera: {e}"

    def analyze_with_model(self, image_path, algorithm, version):
        """Przeprowadza detekcję na obrazie przy użyciu wybranego modelu."""
        model_path = self.get_model_path(algorithm, version)
        if not model_path:
            return f"Błąd: Model {version} dla algorytmu {algorithm} nie istnieje lub nie kończy się na _checkpoint.pth."

        if not os.path.exists(image_path):
            return f"Błąd: Obraz {image_path} nie istnieje."

        # Kopiujemy obraz do folderu data/test/images, aby detect.py mógł go przetworzyć
        test_images_path = self.base_path / "Mask_RCNN" / "data" / "test" / "images"
        test_images_path.mkdir(parents=True, exist_ok=True)
        image_name = os.path.basename(image_path)
        temp_image_path = test_images_path / image_name
        shutil.copy(image_path, temp_image_path)

        # Dostosowujemy ścieżki do kontekstu kontenera
        container_image_path = f"/app/data/test/images/{image_name}"
        container_model_path = f"/app/models/{version}"

        # Uruchamiamy detect.py w kontenerze maskrcnn
        result = self.run_maskrcnn_script("detect.py", container_image_path, container_model_path)
        if "Błąd" in result:
            return result

        # Ścieżka do wyniku detekcji
        result_image_name = os.path.splitext(image_name)[0] + "_detected.jpg"
        result_path = self.detectes_path / result_image_name
        if not result_path.exists():
            return f"Błąd: Wynik detekcji nie został zapisany w {result_path}."

        # Parsowanie liczby detekcji z wyjścia detect.py
        detections_count = 0
        match = re.search(r"Detections: (\d+)", result)
        if match:
            detections_count = int(match.group(1))

        return str(result_path), detections_count
