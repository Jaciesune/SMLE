from PyQt5 import QtWidgets, QtCore, QtGui
from backend.api.train_api import TrainAPI
import os
import time

class TrainingThread(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(str)

    def __init__(self, train_api, algorithm, args):
        super().__init__()
        self.train_api = train_api
        self.algorithm = algorithm
        self.args = args
        self._running = True

    def run(self):
        try:
            for log_line in self.train_api.train_model_stream(self.algorithm, *self.args):
                if not self._running:
                    break
                self.log_signal.emit(log_line)
            self.finished_signal.emit("Trening zakończony sukcesem!")
        except Exception as e:
            self.finished_signal.emit(f"Błąd podczas treningu: {str(e)}")

    def stop(self):
        self._running = False

class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.train_api = TrainAPI()
        self.training_thread = None  # Wątek treningowy
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

        # Przycisk "Rozpocznij trening"
        self.train_btn = QtWidgets.QPushButton("Rozpocznij trening")
        self.train_btn.clicked.connect(self.start_training)
        layout.addRow(self.train_btn)

        # Pole tekstowe do wyświetlania logów
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(200)
        layout.addRow("Logi treningu:", self.log_text)

        form_widget = QtWidgets.QWidget()
        form_widget.setLayout(layout)
        form_widget.setFixedWidth(1000)
        outer_layout.addWidget(form_widget, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(outer_layout)

    def select_dataset(self):
        options = QtWidgets.QFileDialog.Options()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi treningowymi", options=options)
        if folder:
            self.dataset_path_input.setText(folder)

    def select_val_dataset(self):
        options = QtWidgets.QFileDialog.Options()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi walidacyjnymi", options=options)
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

    def start_training(self):
        model_name = self.model_name_input.text().strip()
        train_path = self.dataset_path_input.text().strip()  # Ścieżka do katalogu train użytkownika
        val_path = self.val_path_input.text().strip()  # Ścieżka do katalogu val użytkownika (opcjonalne)
        algorithm = self.algorithm_combo.currentText()
        model_version = self.model_version_combo.currentText()
        epochs = self.epochs_input.value()
        learning_rate = self.learning_rate_input.value()

        # Walidacja
        if not model_name:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę podać nazwę modelu.")
            return
        if not train_path:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać ścieżkę do danych treningowych.")
            return
        if model_version == "Brak dostępnych modeli":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Brak dostępnych modeli do doszkolenia. Wybierz 'Nowy model'.")
            return

        coco_train_filename = "instances_train.json"
        host_coco_train_path = os.path.join(train_path, "annotations", coco_train_filename)
        if not os.path.exists(host_coco_train_path):
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Plik adnotacji treningowych nie istnieje: {host_coco_train_path}")
            return

        # Unikalny identyfikator na podstawie czasu
        timestamp = int(time.time())
        user_train_dir = f"train_user_{timestamp}"

        # Domyślny katalog walidacyjny, jeśli użytkownik nie poda własnego
        if not val_path:
            container_val_path = "/app/backend/data/val"
            user_val_dir = None  # Brak katalogu użytkownika dla val
        else:
            user_val_dir = f"val_user_{timestamp}"
            container_val_path = f"/app/backend/data/{user_val_dir}"

        # Ścieżka treningowa w kontenerze
        container_train_path = f"/app/backend/data/{user_train_dir}"
        coco_val_filename = "instances_val.json"

        # Ścieżki do plików COCO
        coco_train_path = os.path.join(container_train_path, "annotations", coco_train_filename).replace("\\", "/")
        coco_val_filename = "instances_val.json"

        coco_val_path = os.path.join(container_val_path, "annotations", coco_val_filename).replace("\\", "/")

        # Przygotowanie argumentów
        args = [
            "--train_dir", container_train_path,
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--model_name", model_name,
            "--coco_train_path", coco_train_path,
            "--coco_gt_path", coco_val_path,
            "--host_train_path", train_path
        ]
        if val_path:
            args.extend(["--host_val_path", val_path])

        if model_version != "Nowy model":
            model_path = self.train_api.get_model_path(algorithm, model_version)
            if model_path:
                args.extend(["--resume", model_path])
            else:
                QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie można znaleźć modelu: {model_version}")
                return

        self.log_text.clear()
        self.train_btn.setEnabled(False)

        self.training_thread = TrainingThread(self.train_api, algorithm, args)
        self.training_thread.log_signal.connect(self.update_log, QtCore.Qt.QueuedConnection)
        self.training_thread.finished_signal.connect(self.training_finished, QtCore.Qt.QueuedConnection)
        self.training_thread.start()

    def update_log(self, log_line):
        print(f"Debug: Received log: {log_line}")  # Debug
        self.log_text.append(log_line)
        self.log_text.ensureCursorVisible()
        
        # Ograniczenie do np. 1000 linii
        if self.log_text.document().lineCount() > 1000:
            cursor = self.log_text.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.movePosition(QtGui.QTextCursor.Down, QtGui.QTextCursor.KeepAnchor, n=1)
            cursor.removeSelectedText()
        
        QtWidgets.QApplication.processEvents()

    def training_finished(self, result):
        print(f"Debug: Training finished with result: {result}")  # Debug
        self.train_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Trening", result)