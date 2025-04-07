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
            "YOLO": self.base_path / "YOLO" / "models",
            "MCNN": self.base_path / "MCNN" / "models"
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

        return sorted([file.name for file in model_path.iterdir() if file.is_file() and file.name.endswith('_checkpoint.pth')])

    def get_model_path(self, algorithm, version):
        """Zwraca pełną ścieżkę do wybranego modelu."""
        if algorithm not in self.algorithms:
            return None

        model_path = self.algorithms[algorithm] / version
        if model_path.exists() and model_path.is_file() and model_path.name.endswith('_checkpoint.pth'):
            return str(model_path)
        return None

    def run_script(self, script_name, algorithm, *args):
        """Uruchamia skrypt Mask R-CNN w kontenerze maskrcnn."""
        try:
            if algorithm == "Mask R-CNN":
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/Mask_RCNN:/app",
                    "smle-maskrcnn",
                    "python", f"scripts/{script_name}", *args
                ]
            elif algorithm == "MCNN":
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/MCNN:/app/MCNN",
                    "smle-maskrcnn",
                    "python", f"MCNN/{script_name}", *args
                ]
            elif algorithm == "FasterRCNN":
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/FasterRCNN:/app/FasterRCNN",
                    "smle-maskrcnn",
                    "python", f"/app/FasterRCNN/{script_name}", *args
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
            return f"Błąd podczas uruchamiania kontenera: {e}"
        
    def run_fasterrcnn_script(self, script_name, *args):
        try:
            command = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{self.base_path}/FasterRCNN:/app",
                "smle-fasterrcnn",
                "python", script_name, *args
            ]
            print(f"Uruchamiam Faster R-CNN: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Błąd: {result.stderr}"
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Błąd kontenera: {e}"

    def analyze_with_model(self, image_path, algorithm, version):
        """Przeprowadza detekcję na obrazie przy użyciu wybranego modelu."""

        model_path = self.get_model_path(algorithm, version)
        if not model_path:
            return f"Błąd: Model {version} dla {algorithm} nie istnieje."

        if not os.path.exists(image_path):
            return f"Błąd: Obraz {image_path} nie istnieje."

<<<<<<< Updated upstream
        # Kopiujemy obraz do folderu data/test/images, aby detect.py mógł go przetworzyć

        if algorithm == "Mask R-CNN":
            self.detectes_path = self.base_path / "Mask_RCNN" / "data" / "detectes"
            test_images_path = self.base_path / "Mask_RCNN" / "data" / "test" / "images"
            print(test_images_path)
            test_images_path.mkdir(parents=True, exist_ok=True)
            image_name = os.path.basename(image_path)
            temp_image_path = test_images_path / image_name
            print(temp_image_path)
            shutil.copy(image_path, temp_image_path)
            container_image_path = f"/app/data/test/images/{image_name}"
            container_model_path = f"/app/models/{version}"
            result = self.run_script("detect.py", algorithm, container_image_path, container_model_path)

        elif algorithm == "MCNN":
            self.detectes_path = self.base_path / "MCNN" / "data" / "detectes"
            test_images_path = self.base_path / "MCNN" / "data" / "test" / "images"
            print(test_images_path)
            test_images_path.mkdir(parents=True, exist_ok=True)
            image_name = os.path.basename(image_path)
            temp_image_path = test_images_path / image_name
            print(temp_image_path)
            shutil.copy(image_path, temp_image_path)
            container_image_path = f"/app/MCNN/data/test/images/{image_name}"
            container_model_path = f"/app/MCNN/models/{version}"
            result = self.run_script("test_model.py", algorithm, container_image_path, container_model_path)
        
        elif algorithm == "FasterRCNN":
            self.detectes_path = self.base_path / "FasterRCNN" / "data" / "detectes"
            test_images_path = self.base_path / "FasterRCNN" / "data" / "test" / "images"
            print(test_images_path)
            test_images_path.mkdir(parents=True, exist_ok=True)
            image_name = os.path.basename(image_path)
            temp_image_path = test_images_path / image_name
            print(temp_image_path)
            shutil.copy(image_path, temp_image_path)
            container_image_path = f"/app/FasterRCNN/data/test/images/{image_name}"
            container_model_path = f"/app/FasterRCNN/saved_models/{version}"
            result = self.run_script(
                "test.py",
                algorithm,
                "--image_path", container_image_path,
                "--model_path", container_model_path,
                "--output_dir", "/app/FasterRCNN/data/detectes",
                "--threshold", "0.25",
                "--num_classes", "2"
            )
        else:
            return f"Błąd: Algorytm {algorithm} nie jest obsługiwany."

        if "Błąd" in result:
            return result

                # Ścieżka bazowa do folderu backend
        self.detectes_path.mkdir(parents=True, exist_ok=True)

        result_image_name = os.path.splitext(image_name)[0] + "_detected.jpg"
        result_path = self.detectes_path / result_image_name
        if not result_path.exists():
            return f"Błąd: Wynik detekcji nie został zapisany w {result_path}."

        detections_count = 0
        match = re.search(r"Detections: (\d+)", result)
        if match:
            detections_count = int(match.group(1))
=======
        image_name = os.path.basename(image_path)

        if algorithm == "Mask R-CNN":
            test_images_path = self.base_path / "Mask_RCNN" / "data" / "test" / "images"
            test_images_path.mkdir(parents=True, exist_ok=True)
            temp_image_path = test_images_path / image_name
            shutil.copy(image_path, temp_image_path)

            container_image_path = f"/app/data/test/images/{image_name}"
            container_model_path = f"/app/models/{version}"

            result = self.run_maskrcnn_script("detect.py", container_image_path, container_model_path)
            if "Błąd" in result:
                return result

            result_image_name = os.path.splitext(image_name)[0] + "_detected.jpg"
            result_path = self.detectes_path / result_image_name
            if not result_path.exists():
                return f"Błąd: Wynik detekcji nie został zapisany w {result_path}."

            detections_count = 0
            match = re.search(r"Detections: (\d+)", result)
            if match:
                detections_count = int(match.group(1))

            return str(result_path), detections_count

        elif algorithm == "Faster R-CNN":
            test_images_path = self.base_path / "FasterRCNN" / "dataset" / "test"
            test_images_path.mkdir(parents=True, exist_ok=True)
            temp_image_path = test_images_path / image_name
            shutil.copy(image_path, temp_image_path)

            container_image_path = f"/app/dataset/test/{image_name}"
            container_model_path = f"/app/models/{version}"

            result = self.run_fasterrcnn_script("test.py", container_image_path, container_model_path)
            if "Błąd" in result:
                return result

            result_dir = self.base_path / "FasterRCNN" / "test" / "faster_predict"
            result_txt = result_dir / f"{Path(image_name).stem}.txt"
            result_img = result_dir / f"{Path(image_name).stem}.jpg"

            if not result_txt.exists() or not result_img.exists():
                return f"Błąd: Brak plików wynikowych {result_txt} lub {result_img}"

            with open(result_txt, "r") as f:
                count = int(f.readline().strip())

            return str(result_img), count

        else:
            return f"Błąd: Algorytm {algorithm} nie jest obsługiwany."
>>>>>>> Stashed changes

