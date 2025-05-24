"""
Implementacja zakładki Count (Zliczanie) w aplikacji SMLE.

Moduł dostarcza interfejs użytkownika do zliczania obiektów na obrazach
przy użyciu modeli uczenia maszynowego. Umożliwia wybór algorytmu i modelu,
oraz opcjonalne przetwarzanie wstępne obrazów przed analizą.
"""

#######################
# Importy bibliotek
#######################
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import winsound
import requests
import os
import logging
import tempfile
from datetime import datetime

#######################
# Importy lokalne
#######################
from utils import load_stylesheet

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LoadingDialog(QtWidgets.QDialog):
    """
    Okno dialogowe z animacją wyświetlane podczas przetwarzania obrazu.
    
    Zawiera animowany tekst z kropkami, który informuje użytkownika
    o trwającym procesie oznaczania obiektów.
    """
    def __init__(self, parent):
        """
        Inicjalizuje okno dialogowe ładowania.
        Args:
            parent (QtWidgets.QWidget): Widget rodzica (CountTab)
        """
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setObjectName("loading_dialog")
        self.setFixedSize(200, 100)
        self.setStyleSheet("""
            QDialog#loading_dialog {
                background-color: #222831;
                border: 1px solid #948979;
                border-radius: 12px;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 18px;
                font-weight: bold;
            }
        """)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.dots_label = QtWidgets.QLabel("Oznaczanie" + ".")
        self.dots_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.dots_label)
        self.setLayout(layout)

        # Timer do animacji kropek
        self.dot_count = 1
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_dots)
        self.timer.start(500)  # Zmiana co 500 ms

    def update_dots(self):
        """Aktualizuje animację kropek w etykiecie dialogu."""
        self.dot_count = (self.dot_count % 3) + 1
        self.dots_label.setText("Oznaczanie" + "." * self.dot_count)

    def stop(self):
        """Zatrzymuje timer animacji."""
        self.timer.stop()

class CustomDialog(QtWidgets.QDialog):
    """
    Niestandardowy dialog wyświetlający wyniki detekcji obiektów.
    
    Pokazuje nazwę obrazu, liczbę wykrytych obiektów oraz odtwarza
    dźwięk powiadomienia systemowego po zakończeniu analizy.
    """
    def __init__(self, image_name, detections_count, parent, temp_file_path, image_label_size):
        """
        Inicjalizuje dialog wyników detekcji.
        
        Args:
            image_name (str): Nazwa analizowanego obrazu
            detections_count (int): Liczba wykrytych obiektów
            parent (QtWidgets.QWidget): Widget rodzica (CountTab)
            temp_file_path (str): Ścieżka do tymczasowego pliku z przetworzonym obrazem
            image_label_size (QtCore QSize): Rozmiar etykiety wyświetlającej obraz
        """
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setObjectName("count_dialog")
        self.setFixedSize(400, 200)

        # Odtwórz domyślny dźwięk powiadomienia Windows
        winsound.PlaySound("SystemNotification", winsound.SND_ALIAS)

        # Załaduj styl specyficzny dla CountTab_style
        count_tab_stylesheet = load_stylesheet("CountTab_style.css")
        if not count_tab_stylesheet:
            logger.error("[ERROR] Nie udało się wczytać CountTab_style.css")
            count_tab_stylesheet = ""
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
        label3.setAlignment(QtCore.Qt.AlignCenter)

        content_layout.addWidget(label2)
        content_layout.addWidget(label3)

        # Dodanie przycisku OK
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setObjectName("OKButton")
        ok_btn.clicked.connect(self.accept)
        content_layout.addWidget(ok_btn, alignment=QtCore.Qt.AlignCenter)

        main_layout.addWidget(content_widget)
        self.setLayout(main_layout)

        # Wyłącz nakładkę
        parent.overlay_label.setVisible(False)
        QtWidgets.QApplication.processEvents()

        # Podmień obraz w tle
        self.image_update_thread = ImageUpdateThread(temp_file_path, image_label_size)
        self.image_update_thread.image_updated.connect(parent.image_label.setPixmap)
        self.image_update_thread.start()

class ProgressThread(QtCore.QThread):
    """
    Wątek odpowiedzialny za aktualizację animacji dialogu ładowania.
    
    Emituje sygnał stop_progress po zakończeniu pracy.
    """
    stop_progress = QtCore.pyqtSignal()

    def __init__(self, dialog):
        """
        Inicjalizuje wątek aktualizacji animacji dialogu ładowania.
        """
        super().__init__()
        self.dialog = dialog
        self.running = True

    def run(self):
        """Główna pętla wątku, działa dopóki flaga running jest True."""
        while self.running:
            self.msleep(100)

    def stop(self):
        """Zatrzymuje wątek i emituje sygnał zatrzymania."""
        self.running = False
        self.dialog.stop()
        self.stop_progress.emit()

class ImageUpdateThread(QtCore.QThread):
    """
    Wątek aktualizujący wyświetlany obraz po zakończeniu detekcji.
    
    Po zakończeniu pracy usuwa tymczasowy plik obrazu.
    
    Sygnały:
        image_updated: Emitowany z nowym obrazem (QPixmap) po przetworzeniu
    """
    image_updated = QtCore.pyqtSignal(QtGui.QPixmap)

    def __init__(self, temp_file_path, size):
        """
        Inicjalizuje wątek aktualizacji obrazu.
        
        Args:
            temp_file_path (str): Ścieżka do tymczasowego pliku obrazu
            size (QtCore QSize): Docelowy rozmiar obrazu
        """
        super().__init__()
        self.temp_file_path = temp_file_path
        self.size = size

    def run(self):
        """
        Wczytuje obraz z pliku tymczasowego, skaluje go i emituje sygnał z wynikiem.
        Po zakończeniu usuwa plik tymczasowy.
        """
        pixmap = QtGui.QPixmap(self.temp_file_path).scaled(
            self.size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.image_updated.emit(pixmap)
        os.unlink(self.temp_file_path)

class AnalysisThread(QtCore.QThread):
    """
    Wątek wykonujący analizę obrazu w tle.
    
    Komunikuje się z API backendu, przesyła obraz do analizy,
    odbiera przetworzony obraz z detekcjami i liczbą wykrytych obiektów.
    
    Sygnały:
        finished: Emitowany po zakończeniu analizy z wynikiem
        error: Emitowany w przypadku błędu z komunikatem
        show_dialog: Emitowany by wyświetlić dialog z wynikami
    """
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    show_dialog = QtCore.pyqtSignal(str, int, str)

    def __init__(self, image_path, api_url, algorithm, model_version, username, preprocessing_enabled):
        """
        Inicjalizuje wątek analizy obrazu.
        
        Args:
            image_path (str): Ścieżka do pliku obrazu
            api_url (str): Adres URL API backendu
            algorithm (str): Wybrany algorytm detekcji
            model_version (str): Wybrana wersja modelu
            username (str): Nazwa użytkownika wykonującego analizę
            preprocessing_enabled (bool): Czy włączyć preprocessing obrazu
        """
        super().__init__()
        self.image_path = image_path
        self.api_url = api_url
        self.algorithm = algorithm
        self.model_version = model_version
        self.username = username
        self.preprocessing_enabled = preprocessing_enabled

    def run(self):
        """
        Główna metoda wątku wykonująca analizę.
        
        Przesyła obraz do API backendu, odbiera przetworzony obraz
        i informacje o liczbie wykrytych obiektów.
        """
        try:
            logger.debug("Wysyłam żądanie do %s/detect_image: algorithm=%s, model_version=%s, username=%s, preprocessing=%s",
                         self.api_url, self.algorithm, self.model_version, self.username, self.preprocessing_enabled)
            with open(self.image_path, "rb") as image_file:
                files = {"image": (os.path.basename(self.image_path), image_file, "image/jpeg")}
                data = {"algorithm": self.algorithm, "model_version": self.model_version, "username": self.username, "preprocessing": str(self.preprocessing_enabled).lower()}
                response = requests.post(f"{self.api_url}/detect_image", files=files, data=data)
                response.raise_for_status()

            detections_count = int(response.headers.get("X-Detections-Count", 0))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            self.finished.emit({"detections_count": detections_count, "temp_file_path": temp_file_path})
            self.show_dialog.emit(os.path.basename(self.image_path), detections_count, temp_file_path)
        except requests.exceptions.RequestException as e:
            self.error.emit(str(e))

class CountTab(QtWidgets.QWidget):
    """
    Główny komponent zakładki Count (Zliczanie) w aplikacji SMLE.
    
    Umożliwia użytkownikowi wczytanie obrazu, wybór algorytmu i modelu,
    a następnie wykonanie analizy w celu automatycznego zliczenia obiektów.
    Wyniki prezentowane są w formie wizualnej z liczbą wykrytych obiektów.
    """
    def __init__(self, username, api_url):
        """
        Inicjalizuje zakładkę Count.
        
        Args:
            username (str): Nazwa użytkownika korzystającego z aplikacji
            api_url (str): Adres URL API backendu
        """
        super().__init__()
        self.username = username
        self.current_image_path = None
        self.api_url = api_url
        self.analysis_thread = None
        self.progress_thread = None
        self.model_mapping = {}  # Słownik do mapowania wyświetlonych nazw na oryginalne nazwy
        self.init_ui()

    def init_ui(self):
        """
        Tworzy i konfiguruje elementy interfejsu użytkownika zakładki.
        
        Komponenty:
        - Panel widoku obrazu po lewej stronie
        - Panel kontrolny z opcjami po prawej stronie
        - Przyciski do wczytywania obrazów i uruchamiania analizy
        - Pola wyboru algorytmu i modelu
        - Pole wyboru preprocessingu
        """
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        container_widget = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout()

        # Lewa strona - wyświetlanie obrazu
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        image_container = QtWidgets.QWidget()
        image_container.setFixedSize(1000, 800)
        image_container_layout = QtWidgets.QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QtWidgets.QLabel("Tutaj pojawi się zdjęcie")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(1000, 800)
        self.image_label.setStyleSheet("background-color: transparent; border: 1px solid #606060;")

        # Nakładka półprzezroczysta podczas analizy
        self.overlay_label = QtWidgets.QLabel(image_container)
        self.overlay_label.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        self.overlay_label.setVisible(False)
        self.overlay_label.raise_()

        image_container_layout.addWidget(self.image_label)
        left_layout.addWidget(image_container)
        container_layout.addWidget(left_widget)

        # Prawa strona - kontrolki
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setFixedHeight(600)
        right_widget.setFixedWidth(750)
        
        container_layout.addWidget(right_widget)
        container_widget.setLayout(container_layout)
        main_layout.addWidget(container_widget)
        self.setLayout(main_layout)
        right_widget.setLayout(right_layout)
        right_layout.setAlignment(QtCore.Qt.AlignTop)

        # Kontrolki
        self.load_btn = QtWidgets.QPushButton("Wczytaj zdjęcie")
        self.load_btn.clicked.connect(self.load_image)
        right_layout.addWidget(self.load_btn)

        # Wybór algorytmu
        self.algorithm_label = QtWidgets.QLabel("Wybierz algorytm:")
        right_layout.addWidget(self.algorithm_label)
        self.algorithm_combo = QtWidgets.QComboBox()
        self.update_algorithms()
        right_layout.addWidget(self.algorithm_combo)

        # Wybór modelu
        self.model_version_label = QtWidgets.QLabel("Wybierz model:")
        right_layout.addWidget(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        self.algorithm_combo.currentTextChanged.connect(self.update_model_versions)
        self.update_model_versions()
        right_layout.addWidget(self.model_version_combo)

        # Opcje preprocessingu
        self.preprocessing_checkbox = QtWidgets.QCheckBox("Preprocessing")
        self.preprocessing_checkbox.setChecked(False)
        right_layout.addWidget(self.preprocessing_checkbox)

        # Przycisk analizy
        self.analyze_btn = QtWidgets.QPushButton("Rozpocznij analizę")
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.clicked.connect(self.analyze_image)
        right_layout.addWidget(self.analyze_btn)

        # Etykiety na wyniki
        self.image_name_label = QtWidgets.QLabel("")
        self.image_name_label.setVisible(False)
        self.image_name_label.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.image_name_label)

        self.date_label = QtWidgets.QLabel("")
        self.date_label.setVisible(False)
        self.date_label.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.date_label)

        self.detections_count_label = QtWidgets.QLabel("")
        self.detections_count_label.setVisible(False)
        self.detections_count_label.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.detections_count_label)

        # Załaduj style
        count_tab_stylesheet = load_stylesheet("CountTab_style.css")
        if count_tab_stylesheet:
            self.setStyleSheet(count_tab_stylesheet)
            logger.debug("[DEBUG] Załadowano CountTab_style.css")
        else:
            logger.error("[ERROR] Nie udało się wczytać CountTab_style.css")

    def load_image(self):
        """
        Otwiera dialog wyboru pliku obrazu i wyświetla wybrany obraz.
        
        Po wybraniu obrazu z systemu plików, resetuje etykiety wyników
        i ustawia obraz w centralnym widoku.
        """
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
            self.image_label.setStyleSheet("background-color: transparent; border: none;")
            self.image_name_label.setVisible(False)
            self.date_label.setVisible(False)
            self.detections_count_label.setVisible(False)

    def update_algorithms(self):
        """
        Pobiera listę dostępnych algorytmów z API backendu i wypełnia nimi ComboBox.
        
        W przypadku błędu komunikacji wyświetla odpowiedni komunikat.
        """
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
        """
        Pobiera listę dostępnych modeli dla wybranego algorytmu i wypełnia nimi ComboBox.
        
        Wyświetla nazwy bez '_checkpoint.pth', zachowuje oryginalne nazwy w mapowaniu.
        Modele są grupowane w kategorie (Rury, Deski, Książki, Inne).
        Kategorie są pogrubione, a domyślny wybór to pierwszy model w kategorii Rury (jeśli istnieje).
        """
        self.model_version_combo.clear()
        self.model_mapping.clear()  # Wyczyść poprzednie mapowanie
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
                model = QStandardItemModel()
                self.model_version_combo.setModel(model)
                self.model_mapping.clear()
                # Kategorie: pipes, deski, książki, inne
                categorized_models = {"Rury": [], "Deski": [], "Książki": [], "Inne": []}
                for original_model in model_versions:
                    display_model = original_model.replace('_checkpoint.pth', '')
                    prefix = display_model.lower().split('_')[0]  # pierwszy człon przed "_"

                    if prefix == "pipes":
                        categorized_models["Rury"].append((display_model, original_model))
                    elif prefix == "deski":
                        categorized_models["Deski"].append((display_model, original_model))
                    elif prefix == "książki":
                        categorized_models["Książki"].append((display_model, original_model))
                    else:
                        categorized_models["Inne"].append((display_model, original_model))

                # Dodawanie kategorii i modeli
                for category in ["Rury", "Deski", "Książki", "Inne"]:
                    if categorized_models[category]:
                        # Dodaj kategorię jako nieklikalny element z pogrubioną czcionką
                        item = QStandardItem(category)
                        item.setFlags(QtCore.Qt.NoItemFlags)  # Nieklikalny
                        font = QtGui.QFont()
                        font.setBold(True)  # Pogrubienie kategorii
                        item.setFont(font)
                        model.appendRow(item)

                        # Dodaj modele w kategorii
                        for display_model, original_model in categorized_models[category]:
                            entry = QStandardItem(f"  {display_model}")  # Wcięcie dla czytelności
                            entry.setData(original_model, QtCore.Qt.UserRole)
                            model.appendRow(entry)
                            self.model_mapping[display_model] = original_model

                # Ustaw domyślny wybór na pierwszy model w kategorii "Rury", jeśli istnieje
                if categorized_models["Rury"]:
                    # Pierwszy model w kategorii Rury to element zaraz po nazwie kategorii
                    first_rury_index = model.index(1, 0)  # Pierwszy model po kategorii "Rury"
                    self.model_version_combo.setCurrentIndex(first_rury_index.row())
                elif model.rowCount() > 0:
                    # Jeśli brak modeli w kategorii Rury, wybierz pierwszy dostępny model
                    for i in range(model.rowCount()):
                        if model.item(i).flags() & QtCore.Qt.ItemIsSelectable:
                            self.model_version_combo.setCurrentIndex(i)
                            break
            else:
                self.model_version_combo.addItem("Brak dostępnych modeli")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas pobierania modeli: %s", e)
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas pobierania modeli: {e}")
            self.model_version_combo.addItem("Brak dostępnych modeli")

    def analyze_image(self):
        """
        Inicjuje proces analizy obrazu z wybranymi parametrami.
        
        Sprawdza poprawność danych wejściowych, tworzy nakładkę półprzezroczystą
        i dialog ładowania, a następnie uruchamia wątek analizy, który komunikuje
        się z API backendu w celu detekcji obiektów na obrazie.
        """
        if not self.current_image_path:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wczytać zdjęcie.")
            return

        algorithm = self.algorithm_combo.currentText()
        model_version = self.model_version_combo.currentText()
        if not algorithm or algorithm == "Brak dostępnych algorytmów":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać algorytm.")
            return
        if not model_version or model_version in ["pipes", "deski", "książki", "Inne", "Brak dostępnych modeli"]:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać konkretny model, a nie kategorię.")
            return
        
        # Pobierz oryginalną nazwę modelu z mapowania
        original_model_version = self.model_mapping.get(model_version, model_version)
        
        preprocessing_enabled = self.preprocessing_checkbox.isChecked()

        if self.analysis_thread and self.analysis_thread.isRunning():
            logger.warning("Analiza już w toku, zignorowano ponowne kliknięcie.")
            return
        self.analyze_btn.setEnabled(False)
        logger.debug("Aktywuję nakładkę szarą i okno ładowania")

        # Dopasuj overlay_label do rozmiaru i pozycji obrazka
        pixmap = self.image_label.pixmap()
        if (pixmap and not pixmap.isNull()):
            pixmap_size = pixmap.size()
            # Oblicz pozycję, aby overlay był wyśrodkowany na obrazku
            x_offset = (self.image_label.width() - pixmap_size.width()) // 2
            y_offset = (self.image_label.height() - pixmap_size.height()) // 2
            self.overlay_label.setFixedSize(pixmap_size)
            self.overlay_label.move(x_offset, y_offset)
        else:
            # Jeśli brak obrazka, użyj domyślnego rozmiaru (powinno być rzadkie, bo sprawdzamy current_image_path)
            self.overlay_label.setFixedSize(1000, 800)
            self.overlay_label.move(0, 0)

        self.overlay_label.setVisible(True)
        self.overlay_label.raise_()
        QtWidgets.QApplication.processEvents()

        # Pokazanie okna z animacją kropek
        self.loading_dialog = LoadingDialog(self)
        self.loading_dialog.move(
            self.image_label.width() // 2 - self.loading_dialog.width() // 2,
            self.image_label.height() // 2 - self.loading_dialog.height() // 2
        )
        self.loading_dialog.show()

        self.progress_thread = ProgressThread(self.loading_dialog)
        self.progress_thread.start()

        self.analysis_thread = AnalysisThread(
            self.current_image_path, self.api_url, algorithm, original_model_version,
            self.username, preprocessing_enabled
        )
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.show_dialog.connect(self.show_custom_dialog)
        self.analysis_thread.start()

    def show_custom_dialog(self, image_name, detections_count, temp_file_path):
        """
        Wyświetla dialog z wynikami detekcji po zakończeniu analizy.
        
        Args:
            image_name (str): Nazwa analizowanego obrazu
            detections_count (int): Liczba wykrytych obiektów
            temp_file_path (str): Ścieżka do tymczasowego pliku z przetworzonym obrazem
        """
        self.loading_dialog.close()
        dialog = CustomDialog(image_name, detections_count, self, temp_file_path, self.image_label.size())
        dialog.exec_()
        self.analyze_btn.setEnabled(True)

    def on_analysis_finished(self, result):
        """
        Obsługuje zakończenie analizy obrazu.
        
        Aktualizuje etykiety z informacjami o wynikach detekcji.
        
        Args:
            result (dict): Słownik z wynikami analizy zawierający liczbę detekcji
                           i ścieżkę do tymczasowego pliku z przetworzonym obrazem
        """
        logger.debug("Analiza zakończona")
        detections_count = result["detections_count"]
        image_name = os.path.basename(self.current_image_path)
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.image_name_label.setText(f"Nazwa zdjęcia: {image_name}")
        self.date_label.setText(f"Data oznaczenia: {current_date}")
        self.detections_count_label.setText(f"Liczba detekcji: {detections_count}")
        self.image_name_label.setVisible(True)
        self.date_label.setVisible(True)
        self.detections_count_label.setVisible(True)

        if self.progress_thread:
            self.progress_thread.stop()
            self.progress_thread.wait()

    def on_analysis_error(self, error_message):
        """
        Obsługuje błąd podczas analizy obrazu.
        
        Wyświetla komunikat o błędzie, ukrywa etykiety wyników i resetuje stan interfejsu.
        
        Args:
            error_message (str): Komunikat o błędzie
        """
        logger.error("Błąd podczas detekcji: %s", error_message)
        QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas detekcji: {error_message}")
        self.image_name_label.setVisible(False)
        self.date_label.setVisible(False)
        self.detections_count_label.setVisible(False)
        if self.progress_thread:
            self.progress_thread.stop()
            self.progress_thread.wait()
        self.loading_dialog.close()
        self.overlay_label.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QtWidgets.QApplication.processEvents()