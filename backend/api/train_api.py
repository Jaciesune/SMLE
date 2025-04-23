import os
from pathlib import Path
import subprocess
import shutil
import threading
import queue
import requests
import signal
import time

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
        self._running = True
        self._process = None
        self.container_name = "backend-app"

    def get_algorithms(self):
        return list(self.algorithms.keys())

    def get_model_versions(self, algorithm):
        if algorithm not in self.algorithms:
            print(f"Algorytm {algorithm} nie jest wspierany.", flush=True)
            return []
        model_path = self.algorithms[algorithm]
        if not model_path.exists():
            print(f"Katalog modeli {model_path} nie istnieje.", flush=True)
            return []
        model_versions = [file.name for file in model_path.iterdir() if file.is_file() and file.name.endswith('_checkpoint.pth')]
        print(f"Znalezione modele dla {algorithm}: {model_versions}", flush=True)
        return sorted(model_versions)

    def get_model_path(self, algorithm, version):
        if algorithm not in self.algorithms:
            print(f"Algorytm {algorithm} nie jest wspierany.", flush=True)
            return None
        if version.endswith('_checkpoint.pth'):
            print(f"Model wybrany: {version}", flush=True)
            return version
        print(f"Model niepoprawny: {version}", flush=True)
        return None

    def _read_stream(self, stream, q):
        while True:
            try:
                line = stream.readline()
                if not line:
                    break
                q.put((stream, line.strip()))
            except UnicodeDecodeError as e:
                print(f"Błąd dekodowania: {e}")
                q.put((stream, f"[BŁĄD DEKODOWANIA] {str(e)}"))
            except Exception as e:
                print(f"Błąd w _read_stream: {e}")
                break

    def stop(self):
        self._running = False
        if hasattr(self, '_process') and self._process:
            try:
                find_pid_cmd = ["docker", "exec", self.container_name, "ps", "-eo", "pid,cmd", "--no-headers"]
                pid_output = subprocess.check_output(find_pid_cmd, text=True)
                pid = None
                script_name = self.train_scripts.get(self.current_algorithm, "")
                for line in pid_output.splitlines():
                    if script_name in line and "python" in line.lower():
                        pid = line.split()[0]
                        break
                if pid:
                    kill_cmd = ["docker", "exec", self.container_name, "kill", "-9", pid]
                    subprocess.run(kill_cmd, check=True)
                self._process.terminate()
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception as e:
                print(f"Błąd podczas przerywania procesu: {e}")
            finally:
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
                self._process = None

    def train_model_stream(self, algorithm, *args):
        if algorithm not in self.train_scripts:
            yield f"Błąd: Algorytm {algorithm} nie jest wspierany."
            return

        self._running = True
        self.current_algorithm = algorithm
        script_name = self.train_scripts[algorithm]

        train_dir = None
        host_train_path = None
        host_val_path = None
        num_augmentations = "0"
        epochs = "0"
        model_name = ""
        username = ""

        for i in range(0, len(args), 2):
            if args[i] == "--train_dir":
                train_dir = args[i + 1]
            elif args[i] == "--host_train_path":
                host_train_path = args[i + 1]
            elif args[i] == "--host_val_path":
                host_val_path = args[i + 1]
            elif args[i] == "--num_augmentations":
                num_augmentations = args[i + 1]
            elif args[i] == "--epochs":
                epochs = args[i + 1]
            elif args[i] == "--model_name":
                model_name = args[i + 1]
            elif args[i] == "--username":
                self.username = args[i + 1]

        if not train_dir or not host_train_path or not os.path.exists(host_train_path):
            yield f"Błąd: Niepoprawna ścieżka do danych treningowych."
            return

        try:
            host_train_dir = self.base_path / "data" / train_dir.split("/app/backend/data/")[1]
            host_train_dir.mkdir(parents=True, exist_ok=True)

            for item in os.listdir(host_train_path):
                src_path = os.path.join(host_train_path, item)
                dst_path = os.path.join(host_train_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            yield f"Skopiowano dane treningowe do {host_train_dir}"
        except Exception as e:
            yield f"Błąd kopiowania danych treningowych: {e}"
            return

        if host_val_path:
            try:
                host_val_dir = self.base_path / "data" / train_dir.split("/app/backend/data/")[1].replace("train", "val")
                host_val_dir.mkdir(parents=True, exist_ok=True)
                for item in os.listdir(host_val_path):
                    src_path = os.path.join(host_val_path, item)
                    dst_path = os.path.join(host_val_dir, item)
                    if os.path.isfile(src_path):
                        shutil.copy(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                yield f"Skopiowano dane walidacyjne do {host_val_dir}"
            except Exception as e:
                yield f"Błąd kopiowania danych walidacyjnych: {e}"
                return

        def remove_arg_pair(args_list, key):
            args = list(args_list)
            if key in args:
                idx = args.index(key)
                del args[idx:idx+2]
            return args

        # Dodajemy --val_dir do listy argumentów do usunięcia
        filtered_args = list(args)
        for arg_name in ["--host_train_path", "--host_val_path", "--username", "--val_dir"]:
            filtered_args = remove_arg_pair(filtered_args, arg_name)

        try:
            if algorithm == "Mask R-CNN":
                script_path = f"/app/backend/Mask_RCNN/scripts/{script_name}"
            elif algorithm == "MCNN":
                script_path = f"/app/backend/MCNN/{script_name}"
            elif algorithm == "FasterRCNN":
                script_path = f"/app/backend/FasterRCNN/{script_name}"
            else:
                yield f"Błąd: Algorytm {algorithm} nie jest obsługiwany."
                return

            command = ["docker", "exec", "-e", "PYTHONUNBUFFERED=1", self.container_name, "python", script_path]
            command.extend(filtered_args)
            yield f"Uruchamiam trening z {num_augmentations} augmentacjami na obraz..."

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            self._process = process

            output_queue = queue.Queue()
            stdout_thread = threading.Thread(target=self._read_stream, args=(process.stdout, output_queue))
            stderr_thread = threading.Thread(target=self._read_stream, args=(process.stderr, output_queue))
            stdout_thread.start()
            stderr_thread.start()

            while process.poll() is None or stdout_thread.is_alive() or stderr_thread.is_alive():
                if not self._running:
                    self.stop()
                    yield "Trening przerwany przez użytkownika."
                    break
                try:
                    stream, line = output_queue.get(timeout=1.0)
                    if stream == process.stdout and line:
                        yield line
                    elif stream == process.stderr and line:
                        print(f"STDERR: {line}")
                        yield f"[STDERR] {line}"
                except queue.Empty:
                    continue

            stdout_thread.join()
            stderr_thread.join()

            if process.returncode == 0:
                try:
                    requests.post("http://localhost:8000/models/add", json={
                        "name": model_name,
                        "algorithm": algorithm,
                        "path": model_name + "_checkpoint.pth",
                        "epochs": int(epochs),
                        "augmentations": int(num_augmentations),
                        "username": self.username
                    })
                    yield f"Model zapisany do bazy danych przez models_tab"
                except Exception as e:
                    yield f"[OSTRZEŻENIE] Nie udało się wysłać modelu do models_tab: {e}"
            else:
                yield f"Błąd podczas uruchamiania skryptu {script_name}: kod {process.returncode}"
        finally:
            self._process = None