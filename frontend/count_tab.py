from PyQt5 import QtWidgets, QtGui, QtCore
import winsound
import requests
import os
import logging
import tempfile
from datetime import datetime  # Import datetime do pobierania daty
from utils import load_stylesheet  # Import load_stylesheet

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CustomDialog(QtWidgets.QDialog):
    def __init__(self, image_name, detections_count, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setObjectName("count_dialog")
        self.setFixedSize(400, 200)  # Stały rozmiar okna

        # Odtwórz domyślny dźwięk powiadomienia Windows
        winsound.PlaySound("SystemNotification", winsound.SND_ALIAS)

        # Załaduj styl specyficzny dla CountTab_style
        count_tab_stylesheet = load_stylesheet("frontend/styles/CountTab_style.css")
        if not count_tab_stylesheet:
            logger.error("[ERROR] Nie udało się wczytać CountTab_style.css")
            count_tab_stylesheet = ""  # Fallback na pusty styl
        else:
            logger.debug("[DEBUG] Załadowano CountTab_style.css")
        self.setStyleSheet(count_tab_stylesheet)

        # Główny layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Niestandardowy titlebar
        titlebar = QtWidgets.QFrame(self)
        titlebar.setObjectName("TitleBar")
        titlebar_layout = QtWidgets.QHBoxLayout(titlebar)
        titlebar_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QtWidgets.QLabel("Wyniki detekcji")
        title_label.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold; margin-left: 20px; padding: 0px;")
        titlebar_layout.addWidget(title_label)

        main_layout.addWidget(titlebar)

        # Treść dialogu
        content_widget = QtWidgets.QWidget()
        content_widget.setObjectName("ContentWidget")
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(10)

        label2 = QtWidgets.QLabel(f"Nazwa zdjęcia: {image_name}")
        label3 = QtWidgets.QLabel(f"Liczba detekcji: {detections_count}")

        label2.setAlignment(QtCore.Qt.AlignCenter)
        label2.setStyleSheet("color: #FFFFFF; font-size: 18px;")

        label3.setAlignment(QtCore.Qt.AlignCenter)
        label3.setStyleSheet("color: #FFFFFF; font-size: 18px;")

        content_layout.addWidget(label2)
        content_layout.addWidget(label3)

        # Dodanie przycisku OK
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setObjectName("OKButton")
        ok_btn.clicked.connect(self.accept)  # Zamyka okno z kodem akceptacji
        content_layout.addWidget(ok_btn, alignment=QtCore.Qt.AlignCenter)

        main_layout.addWidget(content_widget)
        self.setLayout(main_layout)

        # Usunięcie obsługi przeciągania (niepotrzebne w dialogu informacyjnym)
        self.setFixedPosition(True)

    def setFixedPosition(self, fixed):
        if fixed:
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

class CountTab(QtWidgets.QWidget):
    def __init__(self, username, api_url):  
        super().__init__()
        self.username = username    
        self.current_image_path = None
        self.api_url = api_url  # Przyjmujemy api_url z MainWindow
        self.init_ui()

    def init_ui(self):
        # Główny układ pionowy dla całej zakładki
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Kontener dla lewej i prawej strony (obraz + kontrolki)
        container_widget = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout()

        # Lewa strona - zdjęcie
        left_layout = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel("Tutaj pojawi się zdjęcie")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(1000, 800)  # Rozmiar okna
        self.image_label.setStyleSheet("background-color: transparent; border: 1px solid #606060;")  # Początkowe obramowanie
        left_layout.addWidget(self.image_label)

        # Prawa strona - kontrolki
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setFixedHeight(600)  # Dopasowujemy wysokość do image_label
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

        # Pola z informacjami o detekcji (pod kontrolkami po prawej stronie, w osobnych liniach)
        self.image_name_label = QtWidgets.QLabel("")
        self.image_name_label.setVisible(False)  # Ukryte na starcie
        self.image_name_label.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.image_name_label)

        self.date_label = QtWidgets.QLabel("")
        self.date_label.setVisible(False)  # Ukryte na starcie
        self.date_label.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.date_label)

        self.detections_count_label = QtWidgets.QLabel("")
        self.detections_count_label.setVisible(False)  # Ukryte na starcie
        self.detections_count_label.setStyleSheet("font-size: 16px;")  # Większy font
        right_layout.addWidget(self.detections_count_label)

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
            # Usuwamy obramowanie po wczytaniu zdjęcia
            self.image_label.setStyleSheet("background-color: transparent; border: none;")
            # Po wczytaniu nowego zdjęcia ukrywamy etykiety detekcji
            self.image_name_label.setVisible(False)
            self.date_label.setVisible(False)
            self.detections_count_label.setVisible(False)

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

            detections_count = int(response.headers.get("X-Detections-Count", 0))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            pixmap = QtGui.QPixmap(temp_file_path).scaled(
                self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)

            os.unlink(temp_file_path)

            # Pobierz nazwę zdjęcia i bieżącą datę
            image_name = os.path.basename(self.current_image_path)
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Wyświetlanie informacji w osobnych liniach
            self.image_name_label.setText(f"Nazwa zdjęcia: {image_name}")
            self.date_label.setText(f"Data oznaczenia: {current_date}")
            self.detections_count_label.setText(f"Liczba detekcji: {detections_count}")

            # Pokazujemy etykiety po analizie
            self.image_name_label.setVisible(True)
            self.date_label.setVisible(True)
            self.detections_count_label.setVisible(True)

            # Wyświetlanie niestandardowego dialogu
            dialog = CustomDialog(image_name, detections_count, self)
            dialog.exec_()

        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas detekcji: %s", e)
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas detekcji: {e}")
            # Ukrywamy etykiety w przypadku błędu
            self.image_name_label.setVisible(False)
            self.date_label.setVisible(False)
            self.detections_count_label.setVisible(False)