"""
Implementacja głównego okna aplikacji SMLE (System Maszynowego Liczenia Elementów).

Moduł dostarcza centralny element interfejsu użytkownika, który organizuje wszystkie
funkcjonalności aplikacji w formie zakładek, zapewnia nawigację między nimi
oraz zarządza dostępnością funkcji w zależności od uprawnień użytkownika.
"""

#######################
# Importy bibliotek
#######################
import logging
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtProperty

#######################
# Importy lokalne
#######################
from archive_tab import ArchiveTab
from count_tab import CountTab
from train_tab import TrainTab
from models_tab import ModelsTab
from users_tab import UsersTab
from auto_labeling_tab import AutoLabelingTab
from dataset_creation_tab import DatasetCreationTab
from benchmark_tab import BenchmarkTab  

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CustomToolButton(QtWidgets.QToolButton):
    """
    Niestandardowy przycisk paska narzędzi z obsługą stanu aktywności.
    
    Rozszerza klasę QToolButton o niestandardową właściwość isActive,
    która umożliwia dynamiczną zmianę stylu dla aktywnego przycisku.
    Właściwość jest używana przez arkusz stylów (QSS) do wizualnego
    wyróżnienia aktywnej zakładki.
    """
    def __init__(self, parent=None):
        """
        Inicjalizuje niestandardowy przycisk paska narzędzi.
        
        Args:
            parent (QWidget, optional): Widget rodzica
        """
        super().__init__(parent)
        self._isActive = False

    @pyqtProperty(bool)
    def isActive(self):
        """
        Właściwość Qt określająca, czy przycisk jest aktywny.
        
        Returns:
            bool: Stan aktywności przycisku
        """
        return self._isActive

    @isActive.setter
    def isActive(self, value):
        """
        Ustawia stan aktywności przycisku i odświeża jego styl.
        
        Args:
            value (bool): Nowy stan aktywności
        """
        self._isActive = value
        self.style().unpolish(self)
        self.style().polish(self)

class MainWindow(QtWidgets.QMainWindow):
    """
    Główne okno aplikacji SMLE integrujące wszystkie komponenty interfejsu.
    
    Odpowiada za zarządzanie zakładkami aplikacji, paskiem narzędzi
    do nawigacji między zakładkami oraz warunkowe wyświetlanie
    funkcjonalności w zależności od roli użytkownika.
    """
    def __init__(self, user_role, user_name, stylesheet, api_url):
        """
        Inicjalizuje główne okno aplikacji.
        
        Args:
            user_role (str): Rola użytkownika określająca dostępne funkcje
            user_name (str): Nazwa zalogowanego użytkownika
            stylesheet (str): Zawartość arkusza stylów CSS do zastosowania
            api_url (str): Adres URL API backendu
        """
        super().__init__()
        self.user_role = user_role
        self.username = user_name
        self.selected_folder = None  # Inicjalizacja atrybutu selected_folder
        self.api_url = api_url  # Definiujemy api_url w MainWindow
        logger.debug(f"[DEBUG] Inicjalizacja MainWindow: user_role={self.user_role}, user_name={self.username}, api_url={self.api_url}")
        
        # Ustawienia okna
        self.setWindowTitle("System Maszynowego Liczenia Elementów")
        self.setWindowIcon(QtGui.QIcon("frontend/styles/images/icon.png"))
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(stylesheet)  # Ustaw styl przekazany z main.py
        
        # Inicjalizacja UI
        self.init_ui()
        self.create_toolbar()

    def on_tab_changed(self, index):
        """
        Obsługuje zdarzenie zmiany aktywnej zakładki.
        
        Wykonuje odpowiednie operacje aktualizacji danych
        w zależności od rodzaju wybranej zakładki oraz
        aktualizuje styl aktywnego przycisku w pasku narzędzi.
        
        Args:
            index (int): Indeks nowo wybranej zakładki
        """  
        tab_text = self.tabs.tabText(index)
        logger.debug(f"[DEBUG] Zmiana zakładki na: {tab_text}")
        if tab_text == "Modele":
            self.models_tab.load_models()
        elif tab_text == "Historia":
            self.archive_tab.load_archive_data()
        elif tab_text == "Użytkownicy":
            self.users_tab.load_users()
        elif tab_text == "Benchmark":
            self.benchmark_tab.update_benchmark_results()
        self.update_button_styles(index)  # Aktualizuj styl przycisku dla aktywnej zakładki

    def init_ui(self):
        """
        Tworzy i konfiguruje elementy interfejsu użytkownika głównego okna.
        
        Inicjalizuje wszystkie zakładki aplikacji, konfiguruje widżet QTabWidget
        jako centralny element okna oraz ukrywa domyślny pasek zakładek,
        ponieważ nawigacja odbywa się przez niestandardowy pasek narzędzi.
        Dostępność niektórych zakładek jest zależna od roli użytkownika.
        """
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tworzymy odpowiednie zakładki
        self.count_tab = CountTab(self.username, self.api_url)
        self.train_tab = TrainTab(self.username, self.api_url)
        self.models_tab = ModelsTab(self.username, self.api_url)
        self.archive_tab = ArchiveTab(self.api_url) 
        self.auto_labeling_tab = AutoLabelingTab(self.user_role, self.api_url)
        self.dataset_creation_tab = DatasetCreationTab(self.user_role, self.username, self.api_url)
        self.benchmark_tab = BenchmarkTab(self.user_role, self.api_url)

        self.tabs.addTab(self.count_tab, "Zliczanie")
        self.tabs.addTab(self.train_tab, "Trening")
        self.tabs.addTab(self.models_tab, "Modele")
        self.tabs.addTab(self.auto_labeling_tab, "Półautomatyczne oznaczanie zdjęć")
        self.tabs.addTab(self.dataset_creation_tab, "Tworzenie zbioru danych")
        self.tabs.addTab(self.benchmark_tab, "Benchmark")

        if self.user_role == "admin":
            logger.debug("[DEBUG] Użytkownik jest adminem, dodaję zakładki Użytkownicy i Historia")
            self.users_tab = UsersTab(self.api_url)
            self.tabs.addTab(self.users_tab, "Użytkownicy")
            self.tabs.addTab(self.archive_tab, "Historia")
        else:
            logger.debug("[DEBUG] Użytkownik nie jest adminem, pomijam zakładki Użytkownicy i Historia")

        # Ukryj pasek zakładek, ponieważ przełączanie będzie za pomocą toolbara
        self.tabs.tabBar().hide()

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def create_toolbar(self):
        """
        Tworzy niestandardowy pasek narzędzi do nawigacji między zakładkami.
        
        Pasek zawiera przyciski odpowiadające zakładkom aplikacji oraz
        przycisk wyjścia po prawej stronie. Przyciski wykorzystują klasę
        CustomToolButton, która obsługuje stan aktywności do wizualnego
        wyróżnienia bieżącej zakładki.
        """
        self.toolbar = self.addToolBar("")  # Zapisanie referencji do paska narzędziowego
        self.toolbar.setMovable(False)

        # Lista do przechowywania widgetów przycisków (QToolButton) i akcji
        self.tab_buttons = []
        self.actions = []
        for index in range(self.tabs.count()):
            tab_text = self.tabs.tabText(index)
            action = QtWidgets.QAction(tab_text, self)
            action.triggered.connect(lambda checked, idx=index: self.tabs.setCurrentIndex(idx))
            self.actions.append(action)

            # Tworzenie CustomToolButton i przypisanie akcji
            button = CustomToolButton()
            button.setDefaultAction(action)
            self.toolbar.addWidget(button)
            button.setToolTip("")  # Wyłączenie tooltipów
            button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))  # Ustawienie kursora na rękę wskazującą
            self.tab_buttons.append(button)

            # Dodaj separator po każdej akcji (oprócz ostatniej)
            if index < self.tabs.count() - 1:
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.VLine)
                separator.setStyleSheet("background-color: #000000; width: 2px; height: 25px;")
                self.toolbar.addWidget(separator)

        # Dodajemy separator, aby przesunąć "Wyjście" na prawo
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        # Przycisk "Wyjście" po prawej stronie
        exit_action = QtWidgets.QAction("Wyjście", self)
        exit_action.triggered.connect(self.close)
        self.actions.append(exit_action)

        # Tworzenie CustomToolButton dla "Wyjście"
        self.exit_button = CustomToolButton()  # Zapisanie referencji jako atrybut klasy
        self.exit_button.setDefaultAction(exit_action)
        self.toolbar.addWidget(self.exit_button)
        self.exit_button.setToolTip("")  # Wyłączenie tooltipa dla "Wyjście"
        self.exit_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))  # Ustawienie kursora na rękę wskazującą

        # Początkowe podświetlenie pierwszej zakładki
        self.update_button_styles(0)

    def update_button_styles(self, index):
        """
        Aktualizuje style przycisków paska narzędzi, aby wyróżnić aktywną zakładkę.
        
        Ustawia właściwość isActive na True dla przycisku odpowiadającego
        aktualnie wybranej zakładce i False dla pozostałych przycisków.
        
        Args:
            index (int): Indeks aktywnej zakładki
        """
        # Zaktualizuj styl przycisku dla aktywnej zakładki
        for i, button in enumerate(self.tab_buttons):
            button.isActive = (i == index)  # Ustawiamy isActive dla aktywnej zakładki

        # Przycisk "Wyjście" nie jest zakładką, więc zawsze nieaktywny
        self.exit_button.isActive = False