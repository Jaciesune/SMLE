from PyQt5 import QtWidgets, QtCore

class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Zewnętrzny układ do wycentrowania na ekranie
        outer_layout = QtWidgets.QVBoxLayout()  # Używamy QVBoxLayout dla centrowania pionowego
        outer_layout.setContentsMargins(0, 0, 0, 0)  # Usuwamy marginesy

        # Główny layout
        layout = QtWidgets.QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Nazwa modelu
        self.model_name_input = QtWidgets.QLineEdit()
        layout.addRow("Nazwa Modelu:", self.model_name_input)

        # Ścieżka do danych
        self.dataset_path_input = QtWidgets.QLineEdit()
        self.dataset_path_btn = QtWidgets.QPushButton("Wybierz dane")
        self.dataset_path_btn.clicked.connect(self.select_dataset)
        dataset_layout = QtWidgets.QHBoxLayout()
        dataset_layout.addWidget(self.dataset_path_input)
        dataset_layout.addWidget(self.dataset_path_btn)
        layout.addRow("Ścieżka do danych:", dataset_layout)

        # Lista rozwijana "Wybierz Algorytm"
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItem("CNN")
        self.algorithm_combo.addItem("R-CNN")
        self.algorithm_combo.addItem("Mask R-CNN")

        # Tworzymy layout poziomy dla combo boxa
        algorithm_layout = QtWidgets.QHBoxLayout()
        algorithm_layout.addWidget(self.algorithm_combo)
        layout.addRow("Wybierz Algorytm:", algorithm_layout)

        # Liczba epok
        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(10)
        layout.addRow("Liczba epok:", self.epochs_input)

        # Współczynnik uczenia
        self.learning_rate_input = QtWidgets.QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setValue(0.001)
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

    def start_training(self):
        QtWidgets.QMessageBox.information(self, "Trening", "Rozpoczynam trening modelu...")
