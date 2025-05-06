import json
import requests
import os
import logging
from PyQt5 import QtWidgets
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
    def __init__(self, user_role, user_name):
        super().__init__()
        self.user_role = user_role
        self.username = user_name
        self.selected_folder = None  # Inicjalizacja atrybutu selected_folder
        logger.debug(f"[DEBUG] Inicjalizacja MainWindow: user_role={self.user_role}, user_name={self.username}")
        self.setWindowTitle("System Maszynowego Liczenia Elementów")
        self.setGeometry(100, 100, 1600, 900)
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

    def init_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tworzymy odpowiednie zakładki
        self.count_tab = CountTab(self.username)
        self.train_tab = TrainTab(self.username)
        self.models_tab = ModelsTab()
        self.archive_tab = ArchiveTab()
        self.auto_labeling_tab = AutoLabelingTab(self.user_role)
        self.dataset_creation_tab = DatasetCreationTab(self.user_role, self.username)
        self.benchmark_tab = BenchmarkTab(self.user_role)  # Nowa zakładka

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

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def create_toolbar(self):
        toolbar = self.addToolBar("Główna")
        toolbar.setMovable(False)

        exit_action = QtWidgets.QAction("Wyjście", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)