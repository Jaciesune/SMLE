"""
Implementacja zakładki trenowania modeli w aplikacji SMLE.

Moduł dostarcza interfejs użytkownika do trenowania nowych modeli uczenia maszynowego
oraz doszkalania istniejących modeli. Pozwala na konfigurację parametrów treningu,
wybór algorytmu, monitorowanie postępu oraz zarządzanie procesem treningowym.
"""

#######################
# Importy bibliotek
#######################
from PyQt5 import QtWidgets, QtCore
from pathlib import Path
import uuid
import logging
import os
import re

#######################
# Importy lokalne
#######################
from backend.api.train_api import TrainAPI

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrainingThread(QtCore.QThread):
    """
    Wątek wykonujący operację trenowania modelu w tle.
    
    Odpowiada za komunikację z API trenowania, przekazywanie parametrów
    i streamowanie logów treningu z powrotem do interfejsu użytkownika.
    
    Sygnały:
        log_signal: Emitowany dla każdej nowej linii logu z procesu trenowania
        finished_signal: Emitowany po zakończeniu treningu z informacją o wyniku
    """
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(str)

    def __init__(self, train_api, algorithm, args):
        """
        Inicjalizuje wątek trenowania.
        
        Args:
            train_api (TrainAPI): Instancja API do komunikacji z backendem
            algorithm (str): Nazwa algorytmu do trenowania
            args (list): Argumenty dla procesu trenowania
        """
        super().__init__()
        self.train_api = train_api
        self.algorithm = algorithm
        self.args = args
        self._running = True
        self._stopped_by_user = False

    def run(self):
        """
        Główna metoda wątku uruchamiająca proces trenowania.
        
        Wykonuje trening modelu poprzez API, emituje sygnały z logami
        oraz informacją o zakończeniu procesu lub błędzie.
        """
        try:
            for log_line in self.train_api.train_model_stream(self.algorithm, *self.args):
                if not self._running:
                    break
                self.log_signal.emit(log_line)
            if self._stopped_by_user:
                self.finished_signal.emit("Zatrzymano trening")
            else:
                self.finished_signal.emit("Trening zakończony sukcesem!")
        except Exception as e:
            self.finished_signal.emit(f"Błąd podczas treningu: {str(e)}")

    def stop(self):
        """
        Zatrzymuje proces trenowania.
        
        Ustawia flagę zatrzymania, wywołuje metodę stop() na API
        i czeka na zakończenie wątku.
        """
        self._running = False
        self._stopped_by_user = True
        self.train_api.stop()
        self.wait()  # Czekamy na pełne zakończenie wątku

class TrainTab(QtWidgets.QWidget):
    """
    Zakładka trenowania modeli uczenia maszynowego.
    
    Udostępnia interfejs użytkownika do konfigurowania i uruchamiania
    procesu trenowania modeli, monitorowania postępu oraz przeglądania
    logów. Umożliwia zarówno tworzenie nowych modeli, jak i doszkalanie
    istniejących z wykorzystaniem różnych algorytmów i parametrów.
    """
    def __init__(self, username, api_url):
        """
        Inicjalizuje zakładkę trenowania.
        
        Args:
            username (str): Nazwa użytkownika wykonującego trening
            api_url (str): URL API backendu (obecnie nieużywany bezpośrednio)
        """
        super().__init__()
        self.username = username  
        self.train_api = TrainAPI()
        self.api_url = api_url  # nie używane
        self.training_thread = None
        self.DEFAULT_VAL_PATH = os.getenv("DEFAULT_VAL_PATH", "/app/backend/data/val")
        self.initial_epoch = 0
        self.completed_epochs = 0
        self.total_epochs = 0
        self.user_epochs = 0
        self.init_ui()

    def init_ui(self):
        """
        Tworzy i konfiguruje elementy interfejsu użytkownika zakładki.
        
        Komponenty:
        - Pola konfiguracji modelu (nazwa, ścieżki do danych, algorytm)
        - Wybór modelu do doszkolenia
        - Parametry treningu (liczba epok, współczynnik uczenia, augmentacje)
        - Przyciski sterowania treningiem
        - Pasek postępu i obszar logów
        """
        outer_layout = QtWidgets.QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)

        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Nazwa modelu
        self.model_name_input = QtWidgets.QLineEdit()
        layout.addRow("Nazwa Modelu:", self.model_name_input)

        # Ścieżka do danych treningowych
        self.dataset_path_input = QtWidgets.QLineEdit()
        self.dataset_path_btn = QtWidgets.QPushButton("Wybierz dane treningowe")
        self.dataset_path_btn.clicked.connect(self.select_dataset)
        dataset_layout = QtWidgets.QHBoxLayout()
        dataset_layout.addWidget(self.dataset_path_input)
        dataset_layout.addWidget(self.dataset_path_btn)
        layout.addRow("Ścieżka do danych treningowych:", dataset_layout)

        # Ścieżka do danych walidacyjnych
        self.val_path_input = QtWidgets.QLineEdit()
        self.val_path_btn = QtWidgets.QPushButton("Wybierz dane walidacyjne")
        self.val_path_btn.clicked.connect(self.select_val_dataset)
        val_layout = QtWidgets.QHBoxLayout()
        val_layout.addWidget(self.val_path_input)
        val_layout.addWidget(self.val_path_btn)
        layout.addRow("Ścieżka do danych walidacyjnych:", val_layout)

        # Wybór algorytmu
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItems(self.train_api.get_algorithms())
        self.algorithm_combo.currentTextChanged.connect(self.update_model_versions)
        layout.addRow("Wybierz Algorytm:", self.algorithm_combo)

        # Wybór wersji modelu do doszkolenia
        self.model_version_label = QtWidgets.QLabel("Wybierz model do doszkolenia (opcjonalne):")
        layout.addRow(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        self.model_version_combo.addItem("Nowy model")
        self.update_model_versions()
        layout.addRow(self.model_version_combo)

        # Parametry treningu
        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(10)
        layout.addRow("Liczba epok:", self.epochs_input)

        self.learning_rate_input = QtWidgets.QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setValue(0.0005)
        layout.addRow("Współczynnik uczenia:", self.learning_rate_input)

        self.augmentations_input = QtWidgets.QSpinBox()
        self.augmentations_input.setRange(0, 100)
        self.augmentations_input.setValue(0)
        layout.addRow("Liczba augmentacji na obraz:", self.augmentations_input)

        # Przyciski kontrolne
        self.button_layout = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Rozpocznij trening")
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn = QtWidgets.QPushButton("Zatrzymaj trening")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.button_layout.addWidget(self.train_btn)
        self.button_layout.addWidget(self.stop_btn)
        layout.addRow(self.button_layout)

        # Pasek postępu i logi
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addRow("Postęp:", self.progress_bar)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        self.log_text.setFixedHeight(200)
        layout.addRow("Logi treningu:", self.log_text)

        # Finalizacja layoutu
        form_widget = QtWidgets.QWidget()
        form_widget.setLayout(layout)
        form_widget.setFixedWidth(1000)
        outer_layout.addWidget(form_widget, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(outer_layout)

    def select_dataset(self):
        """
        Otwiera dialog wyboru folderu z danymi treningowymi.
        
        Po wybraniu folderu aktualizuje pole ścieżki danych treningowych.
        """
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi treningowymi")
        if folder:
            self.dataset_path_input.setText(folder)

    def select_val_dataset(self):
        """
        Otwiera dialog wyboru folderu z danymi walidacyjnymi.
        
        Po wybraniu folderu aktualizuje pole ścieżki danych walidacyjnych.
        """
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi walidacyjnymi")
        if folder:
            self.val_path_input.setText(folder)

    def update_model_versions(self):
        """
        Aktualizuje listę dostępnych wersji modeli dla wybranego algorytmu.
        
        Pobiera listę modeli z API trenowania i wypełnia nią combobox,
        dodając zawsze opcję "Nowy model" na początku listy.
        """
        self.model_version_combo.clear()
        self.model_version_combo.addItem("Nowy model")
        algorithm = self.algorithm_combo.currentText()
        model_versions = self.train_api.get_model_versions(algorithm)
        if model_versions:
            self.model_version_combo.addItems(model_versions)
        else:
            self.model_version_combo.addItem("Brak dostępnych modeli")
        logger.info(f"Zaktualizowano listę modeli dla algorytmu {algorithm}: {model_versions}")

    def log_error(self, message):
        """
        Loguje błąd w interfejsie użytkownika i resetuje stan treningu.
        
        Args:
            message (str): Komunikat błędu do wyświetlenia
        """
        self.log_text.appendPlainText(f"Błąd: {message}")
        QtWidgets.QMessageBox.warning(self, "Błąd", message)
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.initial_epoch = 0
        self.completed_epochs = 0
        self.total_epochs = 0
        self.user_epochs = 0
        self.training_thread = None
        logger.info("Stan zresetowany po błędzie.")

    def validate_inputs(self, model_name, train_path, val_path, model_version):
        """
        Weryfikuje poprawność wprowadzonych danych przed rozpoczęciem treningu.
        
        Sprawdza czy podano nazwę modelu, czy ścieżki istnieją i czy wybrano
        prawidłowy model do doszkolenia.
        
        Args:
            model_name (str): Nazwa modelu
            train_path (str): Ścieżka do danych treningowych
            val_path (str): Ścieżka do danych walidacyjnych
            model_version (str): Wybrana wersja modelu
            
        Returns:
            bool: True jeśli wszystkie dane są poprawne, False w przeciwnym przypadku
        """
        if not model_name:
            self.log_error("Proszę podać nazwę modelu.")
            return False
        if not Path(train_path).is_dir():
            self.log_error("Ścieżka do danych treningowych nie jest prawidłowym katalogiem.")
            return False
        if val_path and not Path(val_path).exists():
            self.log_error("Podana ścieżka do danych walidacyjnych nie istnieje.")
            return False
        if model_version == "Brak dostępnych modeli":
            self.log_error("Brak dostępnych modeli do doszkolenia. Wybierz 'Nowy model'.")
            return False
        return True

    def prepare_training_args(self, model_name, train_path, val_path, algorithm, model_version, epochs, lr, augmentations):
        """
        Przygotowuje argumenty dla procesu trenowania.
        
        Tworzy unikalny identyfikator sesji treningowej, konfiguruje ścieżki danych,
        i przygotowuje pełną listę argumentów dla API trenowania.
        
        Args:
            model_name (str): Nazwa modelu
            train_path (str): Ścieżka do danych treningowych
            val_path (str): Ścieżka do danych walidacyjnych
            algorithm (str): Nazwa algorytmu
            model_version (str): Wersja modelu do doszkolenia
            epochs (int): Liczba epok treningu
            lr (float): Współczynnik uczenia
            augmentations (int): Liczba augmentacji na obraz
            
        Returns:
            list: Lista argumentów dla procesu trenowania lub None w przypadku błędu
        """
        unique_id = uuid.uuid4().hex
        user_train_dir = f"train_user_{unique_id}"
        container_train_path = Path("/app/backend/data") / user_train_dir

        coco_train_filename = "instances_train.json"
        host_coco_train_path = Path(train_path) / "annotations" / coco_train_filename
        if not host_coco_train_path.exists():
            self.log_error(f"Plik adnotacji treningowych nie istnieje: {host_coco_train_path}")
            return None

        if not val_path:
            container_val_path = Path(self.DEFAULT_VAL_PATH)
            user_val_dir = None
        else:
            user_val_dir = f"val_user_{unique_id}"
            container_val_path = Path("/app/backend/data") / user_val_dir

        coco_train_path = (container_train_path / "annotations" / coco_train_filename).as_posix()
        coco_val_path = (container_val_path / "annotations" / "instances_val.json").as_posix()

        backend_augmentations = augmentations + 1

        args = [
            "--train_dir", container_train_path.as_posix(),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--model_name", model_name,
            "--coco_train_path", coco_train_path,
            "--coco_gt_path", coco_val_path,
            "--host_train_path", train_path,
            "--num_augmentations", str(backend_augmentations),
            "--username", self.username,
        ]
        if val_path:
            args.extend(["--host_val_path", val_path])
            args.extend(["--val_dir", container_val_path.as_posix()])

        if model_version != "Nowy model":
            logger.info(f"Wybrano model do doszkolenia: {model_version}")
            args.extend(["--resume", model_version])
        else:
            logger.info("Wybrano nowy model, brak --resume")

        logger.info(f"Przygotowane argumenty: {args}")
        return args

    def start_training(self):
        """
        Rozpoczyna proces trenowania modelu.
        
        Pobiera i waliduje parametry z interfejsu użytkownika,
        przygotowuje argumenty, inicjalizuje wątek trenowania
        i aktualizuje stan interfejsu.
        """
        if self.training_thread and self.training_thread.isRunning():
            self.log_error("Trening już trwa!")
            return

        model_name = self.model_name_input.text().strip()
        train_path = self.dataset_path_input.text().strip()
        val_path = self.val_path_input.text().strip()
        algorithm = self.algorithm_combo.currentText()
        model_version = self.model_version_combo.currentText()
        user_epochs = self.epochs_input.value()
        learning_rate = self.learning_rate_input.value()
        augmentations = self.augmentations_input.value()

        logger.info(f"Rozpoczynanie treningu: model_name={model_name}, algorithm={algorithm}, model_version={model_version}")

        if not self.validate_inputs(model_name, train_path, val_path, model_version):
            return

        self.log_text.clear()
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.initial_epoch = 0
        self.completed_epochs = 0
        self.user_epochs = user_epochs
        self.total_epochs = 0

        self.progress_bar.setRange(0, user_epochs)
        self.progress_bar.setValue(0)

        args = self.prepare_training_args(model_name, train_path, val_path, algorithm, model_version, user_epochs, learning_rate, augmentations)
        if args is None:
            return

        self.training_thread = TrainingThread(self.train_api, algorithm, args)
        self.training_thread.log_signal.connect(self.update_log, QtCore.Qt.QueuedConnection)
        self.training_thread.finished_signal.connect(self.training_finished, QtCore.Qt.QueuedConnection)
        self.training_thread.start()
        logger.info(f"Rozpoczęto trening modelu z {augmentations} augmentacjami na obraz (przekazano {augmentations + 1} do backendu).")

    def stop_training(self):
        """
        Zatrzymuje trwający proces trenowania modelu.
        
        Wywołuje metodę stop() na wątku trenowania, aktualizuje
        interfejs i wyświetla informację o zatrzymaniu.
        """
        if self.training_thread and self.training_thread.isRunning():
            logger.info("Rozpoczynanie zatrzymywania treningu...")
            self.training_thread.stop()
            completed = max(0, self.completed_epochs - self.initial_epoch + 1)
            self.log_text.appendPlainText(f"Zatrzymywanie treningu... Ukończono {completed} z {self.user_epochs} epok (od epoki {self.initial_epoch} do {self.completed_epochs}).")
            QtWidgets.QMessageBox.information(self, "Trening", "Zatrzymano trening")
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.train_btn.setEnabled(True)
            logger.info("Zatrzymywanie treningu zakończone.")
        else:
            self.log_text.appendPlainText("Brak aktywnego treningu do zatrzymania.")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.training_thread = None
            logger.info("Próba zatrzymania treningu, ale brak aktywnego wątku.")

    def update_log(self, log_line):
        """
        Aktualizuje obszar logów o nową linię z procesu trenowania.
        
        Analizuje linię logu w poszukiwaniu informacji o aktualnej epoce
        i aktualizuje pasek postępu.
        
        Args:
            log_line (str): Linia logu z procesu trenowania
        """
        logger.debug(f"Otrzymano log: {log_line}")
        self.log_text.appendPlainText(log_line)
        self.log_text.ensureCursorVisible()

        epoch_match = re.search(r"Epoka (\d+)/(\d+)", log_line)
        if not epoch_match:
            epoch_match = re.search(r"Epoka (\d+)\.\.\.", log_line)

        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if self.initial_epoch == 0:
                self.initial_epoch = current_epoch
                self.total_epochs = self.initial_epoch + self.user_epochs - 1
                self.log_text.appendPlainText(f"Docelowa epoka: {self.total_epochs}")
            self.completed_epochs = current_epoch

            epochs_completed_in_session = max(0, self.completed_epochs - self.initial_epoch + 1)
            self.progress_bar.setValue(epochs_completed_in_session)

        QtWidgets.QApplication.processEvents()

    def training_finished(self, result):
        """
        Obsługuje zakończenie procesu trenowania modelu.
        
        Aktualizuje stan interfejsu i wyświetla komunikat o wyniku treningu.
        
        Args:
            result (str): Komunikat o wyniku treningu
        """
        logger.info(f"Trening zakończony: {result}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(self.user_epochs)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.information(self, "Trening", result)
        self.training_thread = None