"""
Moduł API trenowania modeli (Train API)

Ten moduł dostarcza interfejs do zarządzania procesem trenowania różnych modeli
detekcji obiektów (Mask R-CNN, FasterRCNN, MCNN). Umożliwia uruchamianie procesów
treningowych w kontenerach Docker, monitorowanie ich postępu oraz zarządzanie
danymi wejściowymi i wyjściowymi treningu.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na systemie plików
from pathlib import Path  # Do wygodnego zarządzania ścieżkami
import subprocess        # Do uruchamiania procesów zewnętrznych
import shutil            # Do kopiowania plików i katalogów
import threading         # Do obsługi wątków
import queue             # Do komunikacji między wątkami
import requests          # Do komunikacji HTTP
import signal            # Do obsługi sygnałów procesów
import time              # Do operacje związanych z czasem
import logging           # Do logowania informacji i błędów

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TrainAPI:
    """
    Klasa API do trenowania modeli detekcji obiektów.
    
    Dostarcza metody do zarządzania procesem trenowania różnych modeli,
    w tym uruchamiania, monitorowania i zatrzymywania procesu treningu
    oraz zarządzania plikami modelów.
    """
    
    def __init__(self):
        """
        Inicjalizacja API z domyślnymi ścieżkami i ustawieniami.
        
        Konfiguruje ścieżki bazowe dla różnych algorytmów, skrypty treningowe
        oraz domyślne ustawienia procesu treningu.
        """
        self.base_path = Path(__file__).resolve().parent.parent  # backend/
        # Ścieżki katalogów z modelami dla każdego algorytmu
        self.algorithms = {
            "Mask R-CNN": self.base_path / "Mask_RCNN" / "models",
            "FasterRCNN": self.base_path / "FasterRCNN" / "saved_models",
            "MCNN": self.base_path / "MCNN" / "models"
        }
        # Skrypty treningowe dla każdego algorytmu
        self.train_scripts = {
            "Mask R-CNN": "train_maskrcnn.py",
            "FasterRCNN": "run_training.py",
            "MCNN": "train_model.py"
        }
        # Flagi i zmienne stanu
        self._running = True           # Czy proces treningu jest uruchomiony
        self._process = None           # Referencja do procesu treningowego
        self.container_name = "backend-app"  # Nazwa kontenera Docker
        self.current_algorithm = None  # Aktualnie używany algorytm
        self.username = ""             # Nazwa użytkownika inicjującego trening
        
        logger.debug("Zainicjalizowano TrainAPI z bazową ścieżką: %s", self.base_path)

    #######################
    # Metody informacyjne
    #######################

    def get_algorithms(self):
        """
        Zwraca listę dostępnych algorytmów treningowych.
        
        Returns:
            list: Lista nazw dostępnych algorytmów treningowych
        """
        algorithms = list(self.algorithms.keys())
        logger.debug("Pobrano listę algorytmów: %s", algorithms)
        return algorithms

    def get_model_versions(self, algorithm):
        """
        Zwraca listę plików modeli dla wybranego algorytmu.
        
        Parameters:
            algorithm (str): Nazwa algorytmu do sprawdzenia
            
        Returns:
            list: Lista nazw plików modeli dla wybranego algorytmu.
                  Pusta lista, jeśli algorytm nie jest obsługiwany lub katalog nie istnieje.
        """
        if algorithm not in self.algorithms:
            logger.warning("Algorytm %s nie jest wspierany.", algorithm)
            return []
            
        model_path = self.algorithms[algorithm]
        if not model_path.exists():
            logger.warning("Katalog modeli %s nie istnieje.", model_path)
            return []
            
        model_versions = [file.name for file in model_path.iterdir() 
                          if file.is_file() and file.name.endswith('_checkpoint.pth')]
        logger.debug("Znalezione modele dla %s: %s", algorithm, model_versions)
        return sorted(model_versions)

    def get_model_path(self, algorithm, version):
        """
        Zwraca nazwę pliku modelu po weryfikacji jego poprawności.
        
        Parameters:
            algorithm (str): Nazwa algorytmu
            version (str): Nazwa wersji modelu
            
        Returns:
            str: Nazwa pliku modelu, jeśli jest poprawna, lub None w przeciwnym przypadku
        """
        if algorithm not in self.algorithms:
            logger.warning("Algorytm %s nie jest wspierany.", algorithm)
            return None
            
        if version.endswith('_checkpoint.pth'):
            logger.debug("Model wybrany: %s", version)
            return version
            
        logger.warning("Model niepoprawny: %s", version)
        return None

    #######################
    # Zarządzanie procesem treningu
    #######################

    def _read_stream(self, stream, q):
        """
        Wątkowo odczytuje strumień wyjścia procesu i umieszcza linie w kolejce.
        
        Parameters:
            stream: Strumień wyjściowy procesu (stdout lub stderr)
            q (queue.Queue): Kolejka do umieszczania odczytanych linii
        """
        while True:
            try:
                line = stream.readline()
                if not line:
                    break
                q.put((stream, line.strip()))
            except UnicodeDecodeError as e:
                logger.error("Błąd dekodowania: %s", e)
                q.put((stream, f"[BŁĄD DEKODOWANIA] {str(e)}"))
            except Exception as e:
                logger.error("Błąd w _read_stream: %s", e)
                break

    def stop(self):
        """
        Zatrzymuje bieżący proces treningu i czyści zasoby serwera, w tym pamięć VRAM.

        Wysyła SIGTERM do procesu w kontenerze Docker, aby umożliwić skryptowi
        train_maskrcnn.py wykonanie sekcji finally i zwolnienie VRAM. Następnie
        zatrzymuje lokalny proces Popen. Zamyka strumienie stdout i stderr.
        Na koniec usuwa tymczasowe katalogi danych.

        Returns:
            None
        """
        if not self._running:
            logger.warning("Proces już zatrzymany, pomijam zatrzymywanie.")
            return

        self._running = False
        if self._process is not None:
            try:
                logger.info("Próba zakończenia procesu treningu...")
                # Znajdź PID procesu Python uruchomionego w kontenerze
                find_pid_cmd = ["docker", "exec", self.container_name, "ps", "-eo", "pid,cmd", "--no-headers"]
                pid_output = subprocess.check_output(find_pid_cmd, text=True)
                pid = None
                script_name = self.train_scripts.get(self.current_algorithm, "")
                
                for line in pid_output.splitlines():
                    if script_name in line and "python" in line.lower():
                        pid = line.split()[0]
                        break
                
                # Zatrzymaj proces w kontenerze za pomocą SIGTERM
                if pid:
                    logger.debug(f"Znaleziono PID: {pid}, wysyłam SIGTERM do procesu...")
                    term_cmd = ["docker", "exec", self.container_name, "kill", "-SIGTERM", pid]
                    subprocess.run(term_cmd, check=True)
                    
                    # Poczekaj na zakończenie procesu, aby sekcja finally mogła się wykonać
                    logger.debug("Oczekiwanie na zakończenie procesu po SIGTERM...")
                    wait_cmd = ["docker", "exec", self.container_name, "ps", "-p", pid]
                    for _ in range(10):  # Czekaj maksymalnie 10 sekund
                        try:
                            subprocess.run(wait_cmd, check=True, capture_output=True)
                            time.sleep(1)
                        except subprocess.CalledProcessError:
                            logger.debug("Proces zakończony.")
                            break
                    else:
                        logger.warning("Proces nie zakończył się po SIGTERM, wysyłam SIGKILL...")
                        kill_cmd = ["docker", "exec", self.container_name, "kill", "-9", pid]
                        subprocess.run(kill_cmd, check=True)
                
                # Zatrzymaj lokalny proces, jeśli istnieje
                if self._process is not None:
                    logger.debug("Zatrzymywanie lokalnego procesu...")
                    self._process.terminate()
                    self._process.wait(timeout=2)
                    logger.info("Proces zakończony pomyślnie.")
                
            except subprocess.TimeoutExpired:
                logger.warning("Proces nie zakończył się w czasie, wymuszam zamknięcie...")
                if self._process is not None:
                    self._process.kill()
                
            except Exception as e:
                logger.error("Błąd podczas przerywania procesu: %s", e)
                
            finally:
                # Zamknij strumienie wyjściowe
                if self._process is not None:
                    if hasattr(self._process, 'stdout') and self._process.stdout:
                        self._process.stdout.close()
                    if hasattr(self._process, 'stderr') and self._process.stderr:
                        self._process.stderr.close()
                self._process = None
                logger.debug("Proces wyczyszczony, self._process ustawione na None.")
        
        else:
            logger.warning("Brak aktywnego procesu do zatrzymania (self._process jest None).")
        
        # Czyszczenie zasobów systemowych
        try:
            data_dir = self.base_path / "data"
            for subdir in data_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(("train_user_", "val_user_")):
                    logger.info(f"Usuwam tymczasowy katalog: {subdir}")
                    shutil.rmtree(subdir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Błąd podczas usuwania tymczasowych katalogów: {e}")
        
        logger.info("Zasoby systemowe wyczyszczone, proces zatrzymany.")

    #######################
    # Trenowanie modeli
    #######################

    def train_model_stream(self, algorithm, *args):
        """
        Uruchamia proces trenowania modelu i zwraca generator wyjścia.
        
        Kopiuje dane treningowe i walidacyjne do odpowiednich katalogów, 
        uruchamia skrypt treningu w kontenerze Docker i zwraca wyjście
        procesu jako generator. W przypadku błędu rzuca wyjątek.
        
        Parameters:
            algorithm (str): Nazwa algorytmu do trenowania
            *args: Argumenty do przekazania do skryptu treningowego, w formacie:
                   ["--nazwa_argumentu", "wartość", "--inna_nazwa", "inna_wartość", ...]
            
        Yields:
            str: Kolejne linie wyjścia procesu trenowania
            
        Raises:
            RuntimeError: W przypadku błędu podczas trenowania (np. niepowodzenie skryptu)
        """
        if algorithm not in self.train_scripts:
            logger.error(f"Algorytm {algorithm} nie jest wspierany.")
            raise RuntimeError(f"Algorytm {algorithm} nie jest wspierany.")

        self._running = True
        self.current_algorithm = algorithm
        script_name = self.train_scripts[algorithm]

        #######################
        # Parsowanie argumentów
        #######################
        
        train_dir = None
        val_dir = None
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

        logger.debug("Parsowanie argumentów: train_dir=%s, host_train_path=%s, host_val_path=%s, augmentations=%s, epochs=%s, model=%s, user=%s",
                     train_dir, host_train_path, host_val_path, num_augmentations, epochs, model_name, self.username)

        if not train_dir or not host_train_path or not os.path.exists(host_train_path):
            logger.error("Niepoprawna ścieżka do danych treningowych: %s", host_train_path)
            raise RuntimeError("Niepoprawna ścieżka do danych treningowych.")

        #######################
        # Kopiowanie danych treningowych
        #######################
        
        try:
            host_train_dir = self.base_path / "data" / train_dir.split("/app/backend/data/")[1]
            host_train_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Utworzono katalog dla danych treningowych: %s", host_train_dir)

            for item in os.listdir(host_train_path):
                src_path = os.path.join(host_train_path, item)
                dst_path = os.path.join(host_train_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            logger.info("Skopiowano dane treningowe do %s", host_train_dir)
            yield f"Skopiowano dane treningowe do {host_train_dir}"
        except Exception as e:
            logger.error("Błąd kopiowania danych treningowych: %s", e)
            raise RuntimeError(f"Błąd kopiowania danych treningowych: {e}")

        #######################
        # Kopiowanie danych walidacyjnych
        #######################
        
        if host_val_path:
            try:
                host_val_dir = self.base_path / "data" / train_dir.split("/app/backend/data/")[1].replace("train", "val")
                host_val_dir.mkdir(parents=True, exist_ok=True)
                logger.debug("Utworzono katalog dla danych walidacyjnych: %s", host_val_dir)
                
                for item in os.listdir(host_val_path):
                    src_path = os.path.join(host_val_path, item)
                    dst_path = os.path.join(host_val_dir, item)
                    if os.path.isfile(src_path):
                        shutil.copy(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                val_dir = "/app/backend/data/" + str(host_val_dir.relative_to(self.base_path / "data"))
                logger.info("Skopiowano dane walidacyjne do %s", host_val_dir)
                yield f"Skopiowano dane walidacyjne do {host_val_dir}"
            except Exception as e:
                logger.error("Błąd kopiowania danych walidacyjnych: %s", e)
                raise RuntimeError(f"Błąd kopiowania danych walidacyjnych: {e}")
        else:
            logger.warning("Nie podano ścieżki danych walidacyjnych, używam domyślnej")
            val_dir = "/app/backend/data/val"

        #######################
        # Przygotowanie argumentów do treningu
        #######################
        
        def remove_arg_pair(args_list, key):
            args = list(args_list)
            if key in args:
                idx = args.index(key)
                del args[idx:idx+2]
            return args

        filtered_args = list(args)
        for arg_name in ["--host_train_path", "--host_val_path", "--username"]:
            filtered_args = remove_arg_pair(filtered_args, arg_name)
        logger.debug("Argumenty po filtrowaniu: %s", filtered_args)

        if val_dir:
            filtered_args.extend(["--val_dir", val_dir])

        #######################
        # Uruchomienie treningu
        #######################
        
        try:
            if algorithm == "Mask R-CNN":
                script_path = f"/app/backend/Mask_RCNN/scripts/{script_name}"
            elif algorithm == "MCNN":
                script_path = f"/app/backend/MCNN/{script_name}"
            elif algorithm == "FasterRCNN":
                script_path = f"/app/backend/FasterRCNN/{script_name}"
            else:
                logger.error("Algorytm %s nie jest obsługiwany", algorithm)
                raise RuntimeError(f"Algorytm {algorithm} nie jest obsługiwany.")

            command = ["docker", "exec", "-e", "PYTHONUNBUFFERED=1", self.container_name, "python", script_path]
            command.extend(filtered_args)
            logger.info("Uruchamiam trening z %s augmentacjami na obraz...", num_augmentations)
            yield f"Uruchamiam trening z {num_augmentations} augmentacjami na obraz..."
            logger.debug("Komenda: %s", ' '.join(command))
            yield f"Komenda: {' '.join(command)}"

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
            logger.debug("Uruchomiono proces treningowy z PID: %s", process.pid)

            output_queue = queue.Queue()
            stdout_thread = threading.Thread(target=self._read_stream, args=(process.stdout, output_queue))
            stderr_thread = threading.Thread(target=self._read_stream, args=(process.stderr, output_queue))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()
            logger.debug("Uruchomiono wątki do obsługi stdout i stderr")

            while process.poll() is None or not output_queue.empty() or stdout_thread.is_alive() or stderr_thread.is_alive():
                if not self._running:
                    logger.warning("Flaga _running ustawiona na False, przerywam trening")
                    self.stop()
                    yield "Trening przerwany przez użytkownika."
                    break
                try:
                    stream, line = output_queue.get(timeout=1.0)
                    if stream == process.stdout and line:
                        logger.debug("STDOUT: %s", line)
                        yield line
                    elif stream == process.stderr and line:
                        logger.error("STDERR: %s", line)
                        yield f"[STDERR] {line}"
                except queue.Empty:
                    continue

            logger.debug("Oczekiwanie na zakończenie wątków")
            stdout_thread.join()
            stderr_thread.join()
            logger.debug("Wątki zakończone")

            returncode = process.returncode if process.returncode is not None else -1
            logger.info("Proces treningowy zakończony z kodem: %s", returncode)
            
            if returncode != 0:
                logger.error("Błąd podczas uruchamiania skryptu %s: kod %s", script_name, returncode)
                raise RuntimeError(f"Błąd podczas uruchamiania skryptu {script_name}: kod {returncode}")

            try:
                model_data = {
                    "name": model_name,
                    "algorithm": algorithm,
                    "path": model_name + "_checkpoint.pth",
                    "epochs": int(epochs),
                    "augmentations": int(num_augmentations),
                    "username": self.username
                }
                logger.debug("Rejestruję model w bazie danych: %s", model_data)
                
                response = requests.post("http://localhost:8000/models/add", json=model_data)
                
                if response.status_code == 200:
                    logger.info("Model pomyślnie zarejestrowany w bazie danych")
                    yield f"Model zapisany do bazy danych przez models_tab"
                else:
                    logger.warning("Błąd rejestracji modelu: %s %s", response.status_code, response.text)
                    yield f"[OSTRZEŻENIE] Błąd rejestracji modelu: {response.status_code} {response.text}"
            except Exception as e:
                logger.error("Nie udało się wysłać modelu do models_tab: %s", e)
                yield f"[OSTRZEŻENIE] Nie udało się wysłać modelu do models_tab: {e}"
        except Exception as e:
            logger.exception("Nieoczekiwany błąd podczas trenowania modelu: %s", e)
            raise
        finally:
            if self._process and self._process.poll() is None:
                logger.info("Czyszczenie procesu w finally")
                self.stop()
                
            try:
                data_dir = self.base_path / "data"
                for subdir in data_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith(("train_user_", "val_user_")):
                        logger.info(f"Usuwam tymczasowy katalog: {subdir}")
                        shutil.rmtree(subdir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Błąd podczas usuwania tymczasowych katalogów: {e}")