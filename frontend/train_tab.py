from PyQt5 import QtWidgets, QtCore, QtGui
from backend.api.train_api import TrainAPI
from pathlib import Path
import uuid
import time
import logging
import os
import re

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrainingThread(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(str)

    def __init__(self, train_api, algorithm, args):
        super().__init__()
        self.train_api = train_api
        self.algorithm = algorithm
        self.args = args
        self._running = True
        self._stopped_by_user = False  # Flaga określająca, czy trening został wstrzymany przez użytkownika

    def run(self):
        try:
            for log_line in self.train_api.train_model_stream(self.algorithm, *self.args):
                if not self._running:
                    break
                self.log_signal.emit(log_line)
            # Sprawdzamy, czy trening został wstrzymany przez użytkownika
            if self._stopped_by_user:
                self.finished_signal.emit("Zatrzymano trening")
            else:
                self.finished_signal.emit("Trening zakończony sukcesem!")
        except Exception as e:
            self.finished_signal.emit(f"Błąd podczas treningu: {str(e)}")

    def stop(self):
        self._running = False
        self._stopped_by_user = True  # Ustawiamy flagę, że trening został wstrzymany przez użytkownika
        self.train_api.stop()
        self.wait(2000)

class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.train_api = TrainAPI()
        self.training_thread = None
        self.DEFAULT_VAL_PATH = os.getenv("DEFAULT_VAL_PATH", "/app/backend/data/val")
        self.initial_epoch = 0  # Początkowa epoka (dla wznowienia)
        self.completed_epochs = 0  # Licznik ukończonych epok
        self.total_epochs = 0  # Całkowita liczba epok (docelowa)
        self.user_epochs = 0  # Liczba epok podana przez użytkownika
        self.init_ui()

    def init_ui(self):
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

        # Ścieżka do danych walidacyjnych (opcjonalne)
        self.val_path_input = QtWidgets.QLineEdit()
        self.val_path_btn = QtWidgets.QPushButton("Wybierz dane walidacyjne (opcjonalne)")
        self.val_path_btn.clicked.connect(self.select_val_dataset)
        val_layout = QtWidgets.QHBoxLayout()
        val_layout.addWidget(self.val_path_input)
        val_layout.addWidget(self.val_path_btn)
        layout.addRow("Ścieżka do danych walidacyjnych:", val_layout)

        # Lista rozwijana "Wybierz Algorytm"
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItems(self.train_api.get_algorithms())
        self.algorithm_combo.currentTextChanged.connect(self.update_model_versions)
        layout.addRow("Wybierz Algorytm:", self.algorithm_combo)

        # Wybór modelu do doszkolenia
        self.model_version_label = QtWidgets.QLabel("Wybierz model do doszkolenia (opcjonalne):")
        layout.addRow(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        self.model_version_combo.addItem("Nowy model")
        self.update_model_versions()
        layout.addRow(self.model_version_combo)

        # Liczba epok
        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(10)
        layout.addRow("Liczba epok:", self.epochs_input)

        # Współczynnik uczenia
        self.learning_rate_input = QtWidgets.QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setValue(0.0005)
        layout.addRow("Współczynnik uczenia:", self.learning_rate_input)

        # Liczba augmentacji na obraz
        self.augmentations_input = QtWidgets.QSpinBox()
        self.augmentations_input.setRange(0, 100)
        self.augmentations_input.setValue(0)
        layout.addRow("Liczba augmentacji na obraz:", self.augmentations_input)

        # Przyciski "Rozpocznij" i "Zatrzymaj"
        self.button_layout = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Rozpocznij trening")
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn = QtWidgets.QPushButton("Zatrzymaj trening")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.button_layout.addWidget(self.train_btn)
        self.button_layout.addWidget(self.stop_btn)
        layout.addRow(self.button_layout)

        # Pasek postępu
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addRow("Postęp:", self.progress_bar)

        # Pole tekstowe do wyświetlania logów
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        self.log_text.setFixedHeight(200)
        layout.addRow("Logi treningu:", self.log_text)

        form_widget = QtWidgets.QWidget()
        form_widget.setLayout(layout)
        form_widget.setFixedWidth(1000)
        outer_layout.addWidget(form_widget, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(outer_layout)

    def select_dataset(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi treningowymi")
        if folder:
            self.dataset_path_input.setText(folder)

    def select_val_dataset(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi walidacyjnymi")
        if folder:
            self.val_path_input.setText(folder)

    def update_model_versions(self):
        self.model_version_combo.clear()
        self.model_version_combo.addItem("Nowy model")
        algorithm = self.algorithm_combo.currentText()
        model_versions = self.train_api.get_model_versions(algorithm)
        if model_versions:
            self.model_version_combo.addItems(model_versions)
        else:
            self.model_version_combo.addItem("Brak dostępnych modeli")

    def log_error(self, message):
        self.log_text.appendPlainText(f"Błąd: {message}")
        QtWidgets.QMessageBox.warning(self, "Błąd", message)
        # Resetujemy stan po błędzie
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
            "--num_augmentations", str(backend_augmentations)
        ]
        if val_path:
            args.extend(["--host_val_path", val_path])

        if model_version != "Nowy model":
            model_path = self.train_api.get_model_path(algorithm, model_version)
            if model_path:
                args.extend(["--resume", model_path])
            else:
                self.log_error(f"Nie można znaleźć modelu: {model_version}")
                return None

        return args

    def start_training(self):
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
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            completed = max(0, self.completed_epochs - self.initial_epoch + 1)
            self.log_text.appendPlainText(f"Zatrzymywanie treningu... Ukończono {completed} z {self.user_epochs} epok (od epoki {self.initial_epoch} do {self.completed_epochs}).")
            QtWidgets.QMessageBox.information(self, "Trening", "Zatrzymano trening")  # Wyświetlamy okno dialogowe
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.train_btn.setEnabled(True)
            logger.info("Żądanie zatrzymania treningu.")
        else:
            self.log_text.appendPlainText("Brak aktywnego treningu do zatrzymania.")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.training_thread = None

    def update_log(self, log_line):
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
        logger.info(f"Trening zakończony: {result}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(self.user_epochs)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.information(self, "Trening", result)  # Wyświetlamy komunikat zależny od wyniku
        self.training_thread = None