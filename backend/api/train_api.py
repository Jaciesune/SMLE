import os
from pathlib import Path
import subprocess

class TrainAPI:
    def __init__(self):
        # Ścieżka bazowa do folderu backend
        self.base_path = Path(__file__).resolve().parent.parent  # backend/

        # Definicja algorytmów i ich folderów z modelami
        self.algorithms = {
            "Mask R-CNN": self.base_path / "Mask_RCNN" / "models",
            "Faster R-CNN": self.base_path / "Faster_RCNN" / "models",
            "YOLO": self.base_path / "YOLO" / "models"
        }

        # Mapowanie algorytmów na skrypty treningowe
        self.train_scripts = {
            "Mask R-CNN": "train_maskrcnn.py",
            "Faster R-CNN": "train_fasterrcnn.py",  # Do zaimplementowania
            "YOLO": "train_yolo.py"  # Do zaimplementowania
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

    def train_model(self, algorithm, *args):
        """Uruchamia skrypt treningowy w kontenerze smle-maskrcnn."""
        if algorithm not in self.train_scripts:
            return f"Błąd: Algorytm {algorithm} nie jest wspierany."

        script_name = self.train_scripts[algorithm]

        # Parsowanie argumentów, aby znaleźć --train_dir
        train_dir = None
        for i in range(0, len(args), 2):
            if args[i] == "--train_dir":
                train_dir = args[i + 1]
                break

        if not train_dir:
            return f"Błąd: Ścieżka do danych treningowych (--train_dir) nie została podana."

        # Sprawdzenie, czy katalog train_dir istnieje na hoście
        if not os.path.exists(train_dir):
            return f"Błąd: Katalog danych treningowych nie istnieje: {train_dir}"

        try:
            # Przygotowanie polecenia docker run z dynamicznym montowaniem
            command = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{self.base_path}/Mask_RCNN:/app",
                "-v", f"{train_dir}:/app/train_data",
                "--shm-size", "5g",  # Zwiększenie pamięci współdzielonej do 5 GB
                "smle-maskrcnn",
                "python", f"scripts/{script_name}"
            ]

            # Aktualizacja argumentu --train_dir na ścieżkę w kontenerze
            updated_args = []
            i = 0
            while i < len(args):
                if args[i] == "--train_dir":
                    updated_args.append("--train_dir")
                    updated_args.append("/app/train_data")
                    i += 2
                else:
                    updated_args.append(args[i])
                    i += 1

            command.extend(updated_args)
            print(f"Uruchamiam polecenie: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Błąd podczas uruchamiania skryptu {script_name}: {result.stderr}"
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Błąd podczas uruchamiania kontenera: {e}"