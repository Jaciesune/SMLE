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
            "FasterRCNN": self.base_path / "FasterRCNN" / "saved_models",
            "YOLO": self.base_path / "YOLO" / "models",
            "MCNN": self.base_path / "MCNN" / "models"
        }

        # Mapowanie algorytmów na skrypty treningowe
        self.train_scripts = {
            "Mask R-CNN": "train_maskrcnn.py",
            "FasterRCNN": "run_training.py",
            "YOLO": "train_yolo.py",  # Do zaimplementowania
            "MCNN": "train_model.py"
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

        # Parsowanie argumentów, aby znaleźć --train_dir i --host_train_path
        train_dir = None
        host_train_path = None
        for i in range(0, len(args), 2):
            if args[i] == "--train_dir":
                train_dir = args[i + 1]
            elif args[i] == "--host_train_path":
                host_train_path = args[i + 1]

        if not train_dir:
            return f"Błąd: Ścieżka do danych treningowych (--train_dir) nie została podana."

        if not host_train_path:
            return f"Błąd: Ścieżka na hoście (--host_train_path) nie została podana."

        # Sprawdzenie, czy katalog host_train_path istnieje na hoście
        if not os.path.exists(host_train_path):
            return f"Błąd: Katalog danych treningowych nie istnieje: {host_train_path}"

        try:
            if algorithm == "Mask R-CNN":
                # Przygotowanie polecenia docker run z dynamicznym montowaniem
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/Mask_RCNN:/app",
                    "-v", f"{host_train_path}:/data/train",  # Zmiana ścieżki na /dataset/train
                    "--shm-size", "5g",  # Zwiększenie pamięci współdzielonej do 5 GB
                    "smle-maskrcnn",
                    "python", f"scripts/{script_name}"
                ]
            elif algorithm == "MCNN":
                # Przygotowanie polecenia docker run z dynamicznym montowaniem
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/MCNN:/app/MCNN",
                    "-v", f"{host_train_path}:/dataset/train",  # Zmiana ścieżki na /dataset/train
                    "--shm-size", "5g",  # Zwiększenie pamięci współdzielonej do 5 GB
                    "smle-maskrcnn",
                    "python", f"/MCNN/{script_name}"
                ]
            
            elif algorithm == "FasterRCNN":
                # Przygotowanie polecenia docker run z dynamicznym montowaniem
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/FasterRCNN:/app/FasterRCNN",
                    "-v", f"{host_train_path}:/dataset/train",  # Zmiana ścieżki na /dataset/train
                    "-v", f"{self.base_path}/FasterRCNN/dataset/test:/dataset/test",
                    "--shm-size", "5g",  # Zwiększenie pamięci współdzielonej do 5 GB
                    "smle-maskrcnn",
                    "python", f"/app/FasterRCNN/{script_name}"
                ]

            # Usuwamy --host_train_path z argumentów przekazywanych do skryptu
            filtered_args = [arg for arg in args if arg != "--host_train_path" and arg != host_train_path]

            # Zastąpienie ścieżek hosta kontenerowymi ścieżkami wewnątrz dockera
            if algorithm == "MCNN":
                filtered_args = [
                    arg.replace("/data/train", "/dataset/train") if isinstance(arg, str) else arg
                    for arg in filtered_args
                ]

            if algorithm == "FasterRCNN":
                filtered_args = [
                    arg.replace("/data/train", "/dataset/train") if isinstance(arg, str) else arg
                    for arg in filtered_args
                ]

            command.extend(filtered_args)

            print(f"Uruchamiam polecenie: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Błąd podczas uruchamiania skryptu {script_name}: {result.stderr}"
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Błąd podczas uruchamiania kontenera: {e}"

    def train_model_stream(self, algorithm, *args):
        """Uruchamia skrypt treningowy i zwraca logi w czasie rzeczywistym."""
        if algorithm not in self.train_scripts:
            yield f"Błąd: Algorytm {algorithm} nie jest wspierany."
            return

        script_name = self.train_scripts[algorithm]

        # Parsowanie argumentów, aby znaleźć --train_dir i --host_train_path
        train_dir = None
        host_train_path = None
        for i in range(0, len(args), 2):
            if args[i] == "--train_dir":
                train_dir = args[i + 1]
            elif args[i] == "--host_train_path":
                host_train_path = args[i + 1]

        if not train_dir:
            yield f"Błąd: Ścieżka do danych treningowych (--train_dir) nie została podana."
            return

        if not host_train_path:
            yield f"Błąd: Ścieżka na hoście (--host_train_path) nie została podana."
            return

        # Sprawdzenie, czy katalog host_train_path istnieje na hoście
        if not os.path.exists(host_train_path):
            yield f"Błąd: Katalog danych treningowych nie istnieje: {host_train_path}"
            return

        try:
            # Przygotowanie środowiska z PYTHONUNBUFFERED, aby wymusić brak buforowania
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            if algorithm == "Mask R-CNN":
                # Przygotowanie polecenia docker run z dynamicznym montowaniem
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/Mask_RCNN:/app",
                    "-v", f"{host_train_path}:/data/train",  # Zmiana ścieżki na /dataset/train
                    "--shm-size", "5g",  # Zwiększenie pamięci współdzielonej do 5 GB
                    "smle-maskrcnn",
                    "python", f"scripts/{script_name}"
                ]
            
            elif algorithm == "MCNN":
                # Przygotowanie polecenia docker run z dynamicznym montowaniem
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/MCNN:/app/MCNN",
                    "-v", f"{host_train_path}:/dataset/train",  # Zmiana ścieżki na /dataset/train
                    "--shm-size", "5g",  # Zwiększenie pamięci współdzielonej do 5 GB
                    "smle-maskrcnn",
                    "python", f"/MCNN/{script_name}"
                ]
            
            elif algorithm == "FasterRCNN":
                # Przygotowanie polecenia docker run z dynamicznym montowaniem
                command = [
                    "docker", "run", "--rm", "--gpus", "all",
                    "-v", f"{self.base_path}/FasterRCNN:/app/FasterRCNN",
                    "-v", f"{host_train_path}:/dataset/train",  # Zmiana ścieżki na /dataset/train
                    "-v", f"{self.base_path}/FasterRCNN/dataset/test:/dataset/test",
                    "--shm-size", "5g",  # Zwiększenie pamięci współdzielonej do 5 GB
                    "smle-maskrcnn",
                    "python", f"/app/FasterRCNN/{script_name}"
                ]

            # Usuwamy --host_train_path z argumentów przekazywanych do skryptu
            filtered_args = [arg for arg in args if arg != "--host_train_path" and arg != host_train_path]

            # Zastąpienie ścieżek hosta kontenerowymi ścieżkami wewnątrz dockera
            if algorithm == "MCNN":
                filtered_args = [
                    arg.replace("/data/train", "/dataset/train") if isinstance(arg, str) else arg
                    for arg in filtered_args
                ]
            
            if algorithm == "FasterRCNN":
                filtered_args = [
                    arg.replace("/data/train", "/dataset/train") if isinstance(arg, str) else arg
                    for arg in filtered_args
                ]

            command.extend(filtered_args)

            print(f"Uruchamiam polecenie: {' '.join(command)}")
            # Uruchamiamy proces z możliwością strumieniowego odczytu
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Wymuszenie buforowania liniowego
                universal_newlines=True,
                env=env  # Przekazujemy środowisko z PYTHONUNBUFFERED
            )

            # Odczytujemy stdout i stderr w czasie rzeczywistym
            while True:
                stdout_line = process.stdout.readline()
                if stdout_line:
                    yield stdout_line.strip()  # Zwracamy linię stdout

                stderr_line = process.stderr.readline()
                if stderr_line:
                    print(f"STDERR: {stderr_line.strip()}")
                    yield f"[STDERR] {stderr_line.strip()}"

                # Sprawdzamy, czy proces się zakończył
                if process.poll() is not None:
                    break

            # Sprawdzamy kod wyjścia procesu
            if process.returncode != 0:
                yield f"Błąd podczas uruchamiania skryptu {script_name}: proces zakończony z kodem {process.returncode}"
        except subprocess.CalledProcessError as e:
            yield f"Błąd podczas uruchamiania kontenera: {e}"
