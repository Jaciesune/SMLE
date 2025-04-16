from PyQt5 import QtWidgets
from archive_tab import ArchiveTab
from count_tab import CountTab
from train_tab import TrainTab
from models_tab import ModelsTab
from users_tab import UsersTab

from auto_labeling_tab import AutoLabelingTab
from dataset_creation_tab import DatasetCreationTab

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, user_role, user_name):
        super().__init__()
        self.user_role = user_role
        self.username = user_name
        self.setWindowTitle("System Maszynowego Liczenia Elementów")
        self.setGeometry(100, 100, 1600, 900)
        self.init_ui()
        self.create_toolbar()

    def init_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tworzymy odpowiednie zakładki
        self.count_tab = CountTab(self.username)
        self.train_tab = TrainTab()
        self.models_tab = ModelsTab()
        self.archive_tab = ArchiveTab()
        self.auto_labeling_tab = AutoLabelingTab(self.user_role)
        self.dataset_creation_tab = DatasetCreationTab(self.user_role)

        self.tabs.addTab(self.count_tab, "Zliczanie")
        self.tabs.addTab(self.train_tab, "Trening")
        self.tabs.addTab(self.models_tab, "Modele")
        self.tabs.addTab(self.auto_labeling_tab, "Automatyczne oznaczanie zdjęć")
        self.tabs.addTab(self.dataset_creation_tab, "Tworzenie zbioru danych")

        if self.user_role == "admin":
            # Jeśli rola to admin, dodajemy te karty
            self.users_tab = UsersTab()
            self.tabs.addTab(self.users_tab, "Użytkownicy")
            self.tabs.addTab(self.archive_tab, "Historia")

    def create_toolbar(self):
        toolbar = self.addToolBar("Główna")
        toolbar.setMovable(False)

        exit_action = QtWidgets.QAction("Wyjście", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)