import os
from pathlib import Path
import subprocess
import shutil
import re

class DetectionAPI:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent  # backend/
        # Definicja algorytmów i ich folderów z modelami
        self.algorithms = {
            "Mask R-CNN": self.base_path / "Mask_RCNN" / "models",
            "FasterRCNN": self.base_path / "FasterRCNN" / "saved_models",
            "MCNN": self.base_path / "MCNN" / "models",
            "SSD - do zaimplementowania": self.base_path / "SSD" / "models"  # Do zaimplementowania
        }

    def get_algorithms(self):
        """Zwraca listę dostępnych algorytmów."""
        return list(self.algorithms.keys())

    def get_model_versions(self, algorithm):
        """Zwraca listę plików modeli dla wybranego algorytmu. 
        Dla Mask R-CNN tylko pliki z końcówką *_checkpoint.pth, 
        dla reszty wszystkie pliki."""

        if algorithm not in self.algorithms:
            return []

        model_path = self.algorithms[algorithm]
        if not model_path.exists():
            return []

        if algorithm == "Mask R-CNN":
            # Dla Mask R-CNN zwracamy tylko pliki z końcówką *_checkpoint.pth
            return sorted([file.name for file in model_path.iterdir() if file.is_file() and file.name.endswith('_checkpoint.pth')])
        else:
            # Dla pozostałych algorytmów zwracamy wszystkie pliki w katalogu
            return sorted([file.name for file in model_path.iterdir() if file.is_file()])

    def get_model_path(self, algorithm, version):
        """Zwraca pełną ścieżkę do wybranego modelu. 
        Dla Mask R-CNN tylko pliki z końcówką *_checkpoint.pth, 
        dla reszty dowolne pliki."""
        
        if algorithm not in self.algorithms:
            return None

        model_path = self.algorithms[algorithm] / version
        if not model_path.exists() or not model_path.is_file():
            return None

        if algorithm == "Mask R-CNN":
            if model_path.name.endswith('_checkpoint.pth'):
                return str(model_path)
            return None
        else:
            return str(model_path)
        
    def run_script(self, script_name, algorithm, *args):
        """Uruchamia skrypt w już działającym kontenerze backend-app."""
        try:
            # Mapowanie ścieżek dla każdego algorytmu
            if algorithm == "Mask R-CNN":
                script_path = f"/app/backend/Mask_RCNN/scripts/{script_name}"
            elif algorithm == "MCNN":
                script_path = f"/app/backend/MCNN/{script_name}"
            elif algorithm == "FasterRCNN":
                script_path = f"/app/backend/FasterRCNN/{script_name}"
            else:
                return f"Błąd: Algorytm {algorithm} nie jest obsługiwany."

            # Budowanie polecenia Docker exec
            command = [
                "docker", "exec",
                "backend-app",  # Nazwa już działającego kontenera
                "python", script_path, *args
            ]

            print(f"Uruchamiam polecenie: {' '.join(command)}")  # Logowanie dla debugowania
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            if result.returncode != 0:
                return f"Błąd podczas uruchamiania skryptu {script_name}: {result.stderr}"
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Błąd podczas uruchamiania komendy w kontenerze: {e}"
        except Exception as e:
            return f"Nieoczekiwany błąd: {e}"
        
        
    def analyze_with_model(self, image_path, algorithm, version):
        """Przeprowadza detekcję na obrazie przy użyciu wybranego modelu."""
        model_path = self.get_model_path(algorithm, version)
        if not model_path:
            return f"Błąd: Model {version} dla {algorithm} nie istnieje.", 0

        if not os.path.exists(image_path):
            return f"Błąd: Obraz {image_path} nie istnieje.", 0

        # Mapowanie ścieżek dla każdego algorytmu
        if algorithm == "Mask R-CNN":
            host_detectes_path = self.base_path / "Mask_RCNN" / "data" / "detectes"
            host_test_images_path = self.base_path / "Mask_RCNN" / "data" / "test" / "images"
            container_base_path = "/app/backend/Mask_RCNN"
            script_name = "detect.py"
        elif algorithm == "MCNN":
            host_detectes_path = self.base_path / "MCNN" / "data" / "detectes"
            host_test_images_path = self.base_path / "MCNN" / "data" / "test" / "images"
            container_base_path = "/app/backend/MCNN"
            script_name = "test_model.py"
        elif algorithm == "FasterRCNN":
            host_detectes_path = self.base_path / "FasterRCNN" / "data" / "detectes"
            host_test_images_path = self.base_path / "FasterRCNN" / "data" / "test" / "images"
            container_base_path = "/app/backend/FasterRCNN"
            script_name = "test.py"
        else:
            return f"Błąd: Algorytm {algorithm} nie jest obsługiwany.", 0

        # Tworzenie folderów na hoście
        host_test_images_path.mkdir(parents=True, exist_ok=True)
        host_detectes_path.mkdir(parents=True, exist_ok=True)

        # Kopiowanie obrazu na hosta
        image_name = os.path.basename(image_path)
        temp_image_path = host_test_images_path / image_name
        print(f"Kopiuję obraz do: {temp_image_path}")
        try:
            shutil.copy(image_path, temp_image_path)
            if not os.path.exists(temp_image_path):
                return f"Błąd: Nie udało się skopiować obrazu do {temp_image_path}.", 0
            print(f"Obraz został pomyślnie skopiowany do: {temp_image_path}")
        except Exception as e:
            return f"Błąd: Nie udało się skopiować obrazu do {temp_image_path}: {e}", 0

        # Ścieżki w kontenerze
        container_image_path = f"{container_base_path}/data/test/images/{image_name}"
        container_model_path = f"{container_base_path}/{'models' if algorithm != 'FasterRCNN' else 'saved_models'}/{version}"

        # Uruchomienie detekcji
        if algorithm == "FasterRCNN":
            result = self.run_script(
                script_name,
                algorithm,
                "--image_path", container_image_path,
                "--model_path", container_model_path,
                "--output_dir", f"{container_base_path}/data/detectes",
                "--threshold", "0.25",
                "--num_classes", "2"
            )
        else:
            result = self.run_script(script_name, algorithm, container_image_path, container_model_path)

        if "Błąd" in result:
            return result, 0

        # Przetwarzanie wyniku
        result_image_name = os.path.splitext(image_name)[0] + "_detected.jpg"
        result_path = host_detectes_path / result_image_name
        if not result_path.exists():
            return f"Błąd: Wynik detekcji nie został zapisany w {result_path}.", 0

        detections_count = 0
        match = re.search(r"Detections: (\d+)", result)
        if match:
            detections_count = int(match.group(1))

        return str(result_path), detections_count