from PyQt5 import QtWidgets, QtCore
from backend.api.train_api import TrainAPI
import os

class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.train_api = TrainAPI()
        self.init_ui()

    def init_ui(self):
        # Zewnętrzny układ do wycentrowania na ekranie
        outer_layout = QtWidgets.QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Główny layout
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Nazwa modelu
        self.model_name_input = QtWidgets.QLineEdit()
        layout.addRow("Nazwa Modelu:", self.model_name_input)

        # Ścieżka do danych
        self.dataset_path_input = QtWidgets.QLineEdit()
        self.dataset_path_btn = QtWidgets.QPushButton("Wybierz dane treningowe")  # Zmieniamy napis
        self.dataset_path_btn.clicked.connect(self.select_dataset)
        dataset_layout = QtWidgets.QHBoxLayout()
        dataset_layout.addWidget(self.dataset_path_input)
        dataset_layout.addWidget(self.dataset_path_btn)
        layout.addRow("Ścieżka do danych treningowych:", dataset_layout)  # Zmieniamy etykietę

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
        self.update_model_versions()  # Wypełniamy combo box dla domyślnego algorytmu
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

        # Tworzymy widget, który będzie przechowywał nasz layout
        form_widget = QtWidgets.QWidget()
        form_widget.setLayout(layout)

        # Ustawiamy szerokość kontenera na 1000 pikseli
        form_widget.setFixedWidth(1000)

        # Dodajemy form_widget do zewnętrznego layoutu (wyśrodkowanie)
        outer_layout.addWidget(form_widget, alignment=QtCore.Qt.AlignCenter)

        # Ustawiamy zewnętrzny layout
        self.setLayout(outer_layout)

    def select_dataset(self):
        options = QtWidgets.QFileDialog.Options()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z danymi", options=options)
        if folder:
            self.dataset_path_input.setText(folder)

    def update_model_versions(self):
        """Aktualizuje listę wersji modeli na podstawie wybranego algorytmu."""
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
        train_path = self.dataset_path_input.text().strip()  # Ścieżka do katalogu train
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

        # Sprawdzenie, czy plik adnotacji treningowych istnieje (na hoście)
        host_coco_train_path = os.path.join(train_path, "annotations", "coco.json")
        if not os.path.exists(host_coco_train_path):
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Plik adnotacji treningowych nie istnieje: {host_coco_train_path}")
            return

        # Domyślna ścieżka do danych walidacyjnych w kontenerze
        container_val_path = "/app/data/val"
        coco_val_path = os.path.join(container_val_path, "annotations", "coco.json").replace("\\", "/")

        # Przygotowanie argumentów dla train.py
        # Przekazujemy tylko ścieżkę do train, walidacyjna będzie domyślna w train_maskrcnn.py
        args = [
            "--train_dir", train_path,  # Poprawny argument
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--model_name", model_name,
            "--coco_gt_path", coco_val_path
        ]
        if model_version != "Nowy model":
            model_path = self.train_api.get_model_path(algorithm, model_version)
            if model_path:
                args.extend(["--resume", model_path])
            else:
                QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie można znaleźć modelu: {model_version}")
                return

        # Uruchomienie treningu
        QtWidgets.QMessageBox.information(self, "Trening", "Rozpoczynam trening modelu...")
        result = self.train_api.train_model(algorithm, *args)
        if "Błąd" in result:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się przeprowadzić treningu: {result}")
        else:
            QtWidgets.QMessageBox.information(self, "Trening", f"Trening zakończony sukcesem!\nWynik: {result}")