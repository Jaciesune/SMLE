import json
import requests
import os
import logging
from PyQt5 import QtWidgets, QtGui
from archive_tab import ArchiveTab
from count_tab import CountTab
from train_tab import TrainTab
from models_tab import ModelsTab
from users_tab import UsersTab
from auto_labeling_tab import AutoLabelingTab
from dataset_creation_tab import DatasetCreationTab
from benchmark_tab import BenchmarkTab  

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, user_role, user_name, stylesheet):
        super().__init__()
        self.user_role = user_role
        self.username = user_name
        self.selected_folder = None  # Inicjalizacja atrybutu selected_folder
        logger.debug(f"[DEBUG] Inicjalizacja MainWindow: user_role={self.user_role}, user_name={self.username}")
        
        # Ustawienia okna
        self.setWindowTitle("System Maszynowego Liczenia Elementów")
        self.setWindowIcon(QtGui.QIcon("frontend/styles/images/icon.png"))
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(stylesheet)  # Ustaw styl przekazany z main.py
        
        # Inicjalizacja UI
        self.init_ui()
        self.create_toolbar()

    def on_tab_changed(self, index):  
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
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tworzymy odpowiednie zakładki
        self.count_tab = CountTab(self.username)
        self.train_tab = TrainTab(self.username)
        self.models_tab = ModelsTab(self.username)
        self.archive_tab = ArchiveTab()
        self.auto_labeling_tab = AutoLabelingTab(self.user_role)
        self.dataset_creation_tab = DatasetCreationTab(self.user_role, self.username)
        self.benchmark_tab = BenchmarkTab(self.user_role)

        self.tabs.addTab(self.count_tab, "Zliczanie")
        self.tabs.addTab(self.train_tab, "Trening")
        self.tabs.addTab(self.models_tab, "Modele")
        self.tabs.addTab(self.auto_labeling_tab, "Automatyczne oznaczanie zdjęć")
        self.tabs.addTab(self.dataset_creation_tab, "Tworzenie zbioru danych")
        self.tabs.addTab(self.benchmark_tab, "Benchmark")

        if self.user_role == "admin":
            logger.debug("[DEBUG] Użytkownik jest adminem, dodaję zakładki Użytkownicy i Historia")
            self.users_tab = UsersTab()
            self.tabs.addTab(self.users_tab, "Użytkownicy")
            self.tabs.addTab(self.archive_tab, "Historia")
        else:
            logger.debug("[DEBUG] Użytkownik nie jest adminem, pomijam zakładki Użytkownicy i Historia")

        # Ukryj pasek zakładek, ponieważ przełączanie będzie za pomocą toolbara
        self.tabs.tabBar().hide()

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def create_toolbar(self):
        toolbar = self.addToolBar("")
        toolbar.setMovable(False)

        # Lista do przechowywania widgetów przycisków (QToolButton) i akcji
        self.tab_buttons = []
        self.actions = []
        for index in range(self.tabs.count()):
            tab_text = self.tabs.tabText(index)
            action = QtWidgets.QAction(tab_text, self)
            action.triggered.connect(lambda checked, idx=index: self.tabs.setCurrentIndex(idx))
            toolbar.addAction(action)
            self.actions.append(action)

            # Pobierz QToolButton odpowiadający tej akcji
            button = toolbar.widgetForAction(action)
            self.tab_buttons.append(button)

            # Dodaj separator po każdej akcji (oprócz ostatniej)
            if index < self.tabs.count() - 1:
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.VLine)
                separator.setStyleSheet("background-color: #000000; width: 2px; height: 36px;")  # 90% wysokości przycisku (ok. 40px)
                toolbar.addWidget(separator)

        # Dodajemy separator, aby przesunąć "Wyjście" na prawo
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Przycisk "Wyjście" po prawej stronie
        exit_action = QtWidgets.QAction("Wyjście", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)

        # Początkowe podświetlenie pierwszej zakładki
        self.update_button_styles(0)

    def update_button_styles(self, index):
        # Zaktualizuj styl przycisku dla aktywnej zakładki
        for i, button in enumerate(self.tab_buttons):
            if i == index:  # Przycisk aktywnej zakładki
                button.setStyleSheet("background-color: #948979; color: #FFFFFF; border: 1px solid #948979; border-radius: 4px; padding: 8px 15px; min-width: 120px; min-height: 30px;")
            else:
                button.setStyleSheet("background-color: #393E46; color: #FFFFFF; border: none; border-radius: 4px; padding: 8px 15px; min-width: 120px; min-height: 30px;")