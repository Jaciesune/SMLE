import os
from pathlib import Path
import subprocess
import shutil
import threading
import queue

class TrainAPI:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent  # backend/

        self.algorithms = {
            "Mask R-CNN": self.base_path / "Mask_RCNN" / "models",
            "FasterRCNN": self.base_path / "FasterRCNN" / "saved_models",
            "MCNN": self.base_path / "MCNN" / "models",
            "SSD - do zaimplementowania": self.base_path / "SSD" / "models"
        }

        self.train_scripts = {
            "Mask R-CNN": "train_maskrcnn.py",
            "FasterRCNN": "run_training.py",
            "SSD": "train_ssd.py",
            "MCNN": "train_model.py"
        }

    def get_algorithms(self):
        return list(self.algorithms.keys())

    def get_model_versions(self, algorithm):
        if algorithm not in self.algorithms:
            return []
        model_path = self.algorithms[algorithm]
        if not model_path.exists():
            return []
        model_versions = [file.name for file in model_path.iterdir() if file.is_file() and file.name.endswith('_checkpoint.pth')]
        return sorted(model_versions)

    def get_model_path(self, algorithm, version):
        if algorithm not in self.algorithms:
            return None
        model_path = self.algorithms[algorithm] / version
        if model_path.exists() and model_path.is_file() and model_path.name.endswith('_checkpoint.pth'):
            return str(model_path)
        return None

    def _read_stream(self, stream, q):
        """Odczytuje linie ze strumienia i umieszcza je w kolejce."""
        while True:
            line = stream.readline()
            if not line:
                break
            q.put((stream, line.strip()))

    def train_model_stream(self, algorithm, *args):
        if algorithm not in self.train_scripts:
            yield f"Błąd: Algorytm {algorithm} nie jest wspierany."
            return

        script_name = self.train_scripts[algorithm]

        # Parsowanie argumentów
        train_dir = None
        host_train_path = None
        host_val_path = None
        for i in range(0, len(args), 2):
            if args[i] == "--train_dir":
                train_dir = args[i + 1]
            elif args[i] == "--host_train_path":
                host_train_path = args[i + 1]
            elif args[i] == "--host_val_path":
                host_val_path = args[i + 1]

        if not train_dir:
            yield f"Błąd: Ścieżka do danych treningowych (--train_dir) nie została podana."
            return
        if not host_train_path:
            yield f"Błąd: Ścieżka na hoście (--host_train_path) nie została podana."
            return
        if not os.path.exists(host_train_path):
            yield f"Błąd: Katalog danych treningowych nie istnieje: {host_train_path}"
            return
        if host_val_path and not os.path.exists(host_val_path):
            yield f"Błąd: Katalog danych walidacyjnych nie istnieje: {host_val_path}"
            return

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            if algorithm == "Mask R-CNN":
                script_path = f"/app/backend/Mask_RCNN/scripts/{script_name}"
            elif algorithm == "MCNN":
                script_path = f"/app/backend/MCNN/{script_name}"
            elif algorithm == "FasterRCNN":
                script_path = f"/app/backend/FasterRCNN/{script_name}"
            else:
                yield f"Błąd: Algorytm {algorithm} nie jest obsługiwany."
                return

            # Katalog treningowy na hoście
            host_train_dir = self.base_path / "data" / train_dir.split("/app/backend/data/")[1]
            host_train_dir.mkdir(parents=True, exist_ok=True)

            # Kopiowanie danych treningowych
            try:
                for item in os.listdir(host_train_path):
                    src_path = os.path.join(host_train_path, item)
                    dst_path = os.path.join(host_train_dir, item)
                    if os.path.isfile(src_path):
                        shutil.copy(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Skopiowano dane treningowe z {host_train_path} do {host_train_dir}")
            except Exception as e:
                yield f"Błąd: Nie udało się skopiować danych treningowych do {host_train_dir}: {e}"
                return

            # Katalog walidacyjny na hoście (jeśli podano host_val_path)
            if host_val_path:
                host_val_dir = self.base_path / "data" / train_dir.split("/app/backend/data/")[1].replace("train", "val")
                host_val_dir.mkdir(parents=True, exist_ok=True)
                try:
                    for item in os.listdir(host_val_path):
                        src_path = os.path.join(host_val_path, item)
                        dst_path = os.path.join(host_val_dir, item)
                        if os.path.isfile(src_path):
                            shutil.copy(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    print(f"Skopiowano dane walidacyjne z {host_val_path} do {host_val_dir}")
                except Exception as e:
                    yield f"Błąd: Nie udało się skopiować danych walidacyjnych do {host_val_dir}: {e}"
                    return

            # Usuwamy --host_train_path i --host_val_path z argumentów
            filtered_args = [arg for arg in args if arg not in ["--host_train_path", host_train_path, "--host_val_path", host_val_path]]

            command = [
                "docker", "exec",
                "-e", "PYTHONUNBUFFERED=1",
                "backend-app",
                "python", script_path
            ]
            command.extend(filtered_args)

            print(f"Uruchamiam polecenie: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            # Kolejka do przechowywania linii z wątków
            output_queue = queue.Queue()

            # Uruchomienie wątków do odczytu stdout i stderr
            stdout_thread = threading.Thread(target=self._read_stream, args=(process.stdout, output_queue))
            stderr_thread = threading.Thread(target=self._read_stream, args=(process.stderr, output_queue))
            stdout_thread.start()
            stderr_thread.start()

            # Odczyt z kolejki, dopóki proces się nie zakończy i wątki są aktywne
            while process.poll() is None or stdout_thread.is_alive() or stderr_thread.is_alive():
                try:
                    # Odczyt z kolejki z timeoutem, aby nie blokować
                    stream, line = output_queue.get(timeout=1.0)
                    if stream == process.stdout and line:
                        yield line
                    elif stream == process.stderr and line:
                        print(f"STDERR: {line}")
                        yield f"[STDERR] {line}"
                except queue.Empty:
                    continue  # Jeśli kolejka jest pusta, kontynuuj pętlę

            # Czekaj na zakończenie wątków
            stdout_thread.join()
            stderr_thread.join()

            # Sprawdzenie kodu zakończenia
            if process.returncode != 0:
                yield f"Błąd podczas uruchamiania skryptu {script_name}: proces zakończony z kodem {process.returncode}"

        except subprocess.CalledProcessError as e:
            yield f"Błąd podczas uruchamiania komendy w kontenerze: {e}"
        except Exception as e:
            yield f"Nieoczekiwany błąd: {e}"
