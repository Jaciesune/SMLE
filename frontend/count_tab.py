from PyQt5 import QtWidgets, QtGui, QtCore
import requests
import os
import logging
import tempfile

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CountTab(QtWidgets.QWidget):
    def __init__(self, username):  
        super().__init__()
        self.username = username    
        self.current_image_path = None
        self.api_url = "http://localhost:8000"
        self.init_ui()


    def init_ui(self):
        # Główny układ poziomy
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Kontener dla obu stron
        container_widget = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout()

        # Lewa strona - zdjęcie
        left_layout = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel("Tutaj pojawi się zdjęcie")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(500, 400)
        self.image_label.setStyleSheet("border: 1px solid #606060; background-color: #767676;")
        left_layout.addWidget(self.image_label)

        # Prawa strona - kontrolki
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setFixedHeight(400)
        right_widget.setFixedWidth(750)
        
        container_layout.addLayout(left_layout)
        container_layout.addWidget(right_widget)

        container_widget.setLayout(container_layout)
        main_layout.addWidget(container_widget)

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
        self.update_algorithms()
        right_layout.addWidget(self.algorithm_combo)

        # Wybór wersji modelu
        self.model_version_label = QtWidgets.QLabel("Wybierz model:")
        right_layout.addWidget(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        self.algorithm_combo.currentTextChanged.connect(self.update_model_versions)
        self.update_model_versions()
        right_layout.addWidget(self.model_version_combo)

        # Checkbox "Preprocessing"
        self.preprocessing_checkbox = QtWidgets.QCheckBox("Preprocessing")
        self.preprocessing_checkbox.setChecked(False)  # Domyślnie niezaznaczony
        right_layout.addWidget(self.preprocessing_checkbox)


        # Przycisk "Rozpocznij analizę"
        self.analyze_btn = QtWidgets.QPushButton("Rozpocznij analizę")
        self.analyze_btn.clicked.connect(self.analyze_image)
        right_layout.addWidget(self.analyze_btn)

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Wybierz zdjęcie", "", "Obrazy (*.png *.jpg *.jpeg *.bmp)", options=options
        )
        if file_path:
            self.current_image_path = file_path
            pixmap = QtGui.QPixmap(file_path).scaled(
                self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)

    def update_algorithms(self):
        """Pobiera listę dostępnych algorytmów z API."""
        try:
            logger.debug("Pobieram listę algorytmów z %s/detect_algorithms", self.api_url)
            response = requests.get(f"{self.api_url}/detect_algorithms")
            response.raise_for_status()
            algorithms = response.json()
            self.algorithm_combo.clear()
            if algorithms:
                self.algorithm_combo.addItems(algorithms)
            else:
                self.algorithm_combo.addItem("Brak dostępnych algorytmów")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas pobierania algorytmów: %s", e)
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas pobierania algorytmów: {e}")
            self.algorithm_combo.clear()
            self.algorithm_combo.addItem("Brak dostępnych algorytmów")

    def update_model_versions(self):
        """Aktualizuje listę wersji modeli na podstawie wybranego algorytmu."""
        self.model_version_combo.clear()
        algorithm = self.algorithm_combo.currentText()
        if not algorithm or algorithm == "Brak dostępnych algorytmów":
            self.model_version_combo.addItem("Brak dostępnych modeli")
            return
        try:
            logger.debug("Pobieram modele dla algorytmu %s z %s/detect_model_versions/%s", 
                        algorithm, self.api_url, algorithm)
            response = requests.get(f"{self.api_url}/detect_model_versions/{algorithm}")
            response.raise_for_status()
            model_versions = response.json()
            if model_versions:
                self.model_version_combo.addItems(model_versions)
            else:
                self.model_version_combo.addItem("Brak dostępnych modeli")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas pobierania modeli: %s", e)
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas pobierania modeli: {e}")
            self.model_version_combo.addItem("Brak dostępnych modeli")

    def analyze_image(self):
        if not self.current_image_path:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wczytać zdjęcie.")
            return

        algorithm = self.algorithm_combo.currentText()
        model_version = self.model_version_combo.currentText()
        if not algorithm or algorithm == "Brak dostępnych algorytmów":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać algorytm.")
            return
        if not model_version or model_version == "Brak dostępnych modeli":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać model.")
            return
        
         # Pobierz stan checkboxa Preprocessing
        preprocessing_enabled = self.preprocessing_checkbox.isChecked()

        try:
            logger.debug("Wysyłam żądanie do %s/detect_image: algorithm=%s, model_version=%s, username=%s, preprocessing=%s",
                         self.api_url, algorithm, model_version, self.username, preprocessing_enabled)
            with open(self.current_image_path, "rb") as image_file:
                files = {"image": (os.path.basename(self.current_image_path), image_file, "image/jpeg")}
                data = {"algorithm": algorithm, "model_version": model_version, "username": self.username, "preprocessing": str(preprocessing_enabled).lower()}
                response = requests.post(f"{self.api_url}/detect_image", files=files, data=data)
                response.raise_for_status()

            # Pobranie liczby detekcji z nagłówka
            detections_count = int(response.headers.get("X-Detections-Count", 0))

            # Zapisanie obrazu wynikowego tymczasowo
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            # Wyświetlenie obrazu wynikowego
            pixmap = QtGui.QPixmap(temp_file_path).scaled(
                self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)

            # Usunięcie pliku tymczasowego
            os.unlink(temp_file_path)

            QtWidgets.QMessageBox.information(
                self, "Analiza", f"Detekcja zakończona. Liczba detekcji: {detections_count}."
            )
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas detekcji: %s", e)
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas detekcji: {e}")