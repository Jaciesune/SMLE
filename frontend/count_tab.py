from PyQt5 import QtWidgets, QtGui, QtCore
from backend.api.detection_api import DetectionAPI

class CountTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.current_image_path = None  # Przechowujemy ścieżkę do wczytanego obrazu
        self.detection_api = DetectionAPI()  # Inicjalizacja DetectionAPI
        self.init_ui()

    def init_ui(self):
        # Główny układ poziomy
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Tworzymy kontener dla obu stron, wycentrowany na ekranie
        container_widget = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout()

        # Układ dla lewej strony - zdjęcie
        left_layout = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel("Tutaj pojawi się zdjęcie")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(500, 400)
        self.image_label.setStyleSheet("border: 1px solid #606060; background-color: #767676;")
        left_layout.addWidget(self.image_label)

        # Tworzymy kontener dla prawej strony
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setFixedHeight(400)
        right_widget.setFixedWidth(750)
        
        # Dodajemy lewą i prawą część do głównego układu
        container_layout.addLayout(left_layout)
        container_layout.addWidget(right_widget)

        # Ustawiamy kontener na środku ekranu
        container_widget.setLayout(container_layout)
        main_layout.addWidget(container_widget)

        # Ustawiamy główny układ
        self.setLayout(main_layout)
        right_widget.setLayout(right_layout)
        right_layout.setAlignment(QtCore.Qt.AlignTop)

        # Przycisk "Wczytaj zdjęcie"
        self.load_btn = QtWidgets.QPushButton("Wczytaj zdjęcie")
        self.load_btn.clicked.connect(self.load_image)
        right_layout.addWidget(self.load_btn)

        # Wybór algorytmu
        self.algorithm_label = QtWidgets.QLabel("Wybierz algorytm:")
        right_layout.addWidget(self.algorithm_label)
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItems(self.detection_api.get_algorithms())
        self.algorithm_combo.currentTextChanged.connect(self.update_model_versions)
        right_layout.addWidget(self.algorithm_combo)

        # Wybór wersji modelu
        self.model_version_label = QtWidgets.QLabel("Wybierz model:")
        right_layout.addWidget(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        self.update_model_versions()  # Wypełniamy combo box dla domyślnego algorytmu
        right_layout.addWidget(self.model_version_combo)

        # Przycisk "Rozpocznij analizę"
        self.analyze_btn = QtWidgets.QPushButton("Rozpocznij analizę")
        self.analyze_btn.clicked.connect(self.analyze_image)
        right_layout.addWidget(self.analyze_btn)

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz zdjęcie", "", "Obrazy (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.current_image_path = file_path
            pixmap = QtGui.QPixmap(file_path).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

    def update_model_versions(self):
        """Aktualizuje listę wersji modeli na podstawie wybranego algorytmu."""
        self.model_version_combo.clear()
        algorithm = self.algorithm_combo.currentText()
        model_versions = self.detection_api.get_model_versions(algorithm)
        if model_versions:
            self.model_version_combo.addItems(model_versions)
        else:
            self.model_version_combo.addItem("Brak dostępnych modeli")

    def analyze_image(self):
        if not self.current_image_path:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wczytać zdjęcie.")
            return

        algorithm = self.algorithm_combo.currentText()
        model_version = self.model_version_combo.currentText()
        if not model_version or model_version == "Brak dostępnych modeli":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać model.")
            return

        try:
            result = self.detection_api.analyze_with_model(self.current_image_path, algorithm, model_version)
            if isinstance(result, str) and "Błąd" in result:
                QtWidgets.QMessageBox.warning(self, "Błąd", result)
                return

            result_path, detections_count = result  # Rozpakowanie dwóch wartości
            pixmap = QtGui.QPixmap(result_path).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            QtWidgets.QMessageBox.information(self, "Analiza", f"Detekcja zakończona. Liczba detekcji: {detections_count}. Wynik zapisano w: {result_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się przeprowadzić detekcji: {str(e)}")