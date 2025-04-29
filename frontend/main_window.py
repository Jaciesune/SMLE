from PyQt5 import QtWidgets
from archive_tab import ArchiveTab
from count_tab import CountTab
from train_tab import TrainTab
from models_tab import ModelsTab
from users_tab import UsersTab
from auto_labeling_tab import AutoLabelingTab
from dataset_creation_tab import DatasetCreationTab
import requests
import os
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QComboBox

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, user_role, user_name):
        super().__init__()
        self.user_role = user_role
        self.username = user_name
        self.selected_folder = None  # Inicjalizacja atrybutu selected_folder
        print(f"[DEBUG] Inicjalizacja MainWindow: user_role={self.user_role}, user_name={self.username}")
        self.setWindowTitle("System Maszynowego Liczenia Elementów")
        self.setGeometry(100, 100, 1600, 900)
        self.init_ui()
        self.create_toolbar()

    def on_tab_changed(self, index):  
        tab_text = self.tabs.tabText(index)
        print(f"[DEBUG] Zmiana zakładki na: {tab_text}")
        if tab_text == "Modele":
            self.models_tab.load_models()
        elif tab_text == "Historia":
            self.archive_tab.load_archive_data()
        elif tab_text == "Użytkownicy":
            self.users_tab.load_users()
        elif tab_text == "Benchmark":
            self.update_benchmark_results()

    def init_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tworzymy odpowiednie zakładki
        self.count_tab = CountTab(self.username)
        self.train_tab = TrainTab(self.username)
        self.models_tab = ModelsTab()
        self.archive_tab = ArchiveTab()
        self.auto_labeling_tab = AutoLabelingTab(self.user_role)
        self.dataset_creation_tab = DatasetCreationTab(self.user_role)

        self.tabs.addTab(self.count_tab, "Zliczanie")
        self.tabs.addTab(self.train_tab, "Trening")
        self.tabs.addTab(self.models_tab, "Modele")
        self.tabs.addTab(self.auto_labeling_tab, "Automatyczne oznaczanie zdjęć")
        self.tabs.addTab(self.dataset_creation_tab, "Tworzenie zbioru danych")
        
        # Dodajemy zakładkę Benchmark
        self.benchmark_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.benchmark_tab, "Benchmark")
        self.setup_benchmark_tab()

        if self.user_role == "admin":
            print("[DEBUG] Użytkownik jest adminem, dodaję zakładki Użytkownicy i Historia")
            self.users_tab = UsersTab()
            self.tabs.addTab(self.users_tab, "Użytkownicy")
            self.tabs.addTab(self.archive_tab, "Historia")
        else:
            print("[DEBUG] Użytkownik nie jest adminem, pomijam zakładki Użytkownicy i Historia")

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def create_toolbar(self):
        toolbar = self.addToolBar("Główna")
        toolbar.setMovable(False)

        exit_action = QtWidgets.QAction("Wyjście", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)

    def setup_benchmark_tab(self):
        print("[DEBUG] Konfiguracja zakładki Benchmark")
        layout = QVBoxLayout()
        
        # Etykieta wyników
        self.benchmark_results_label = QLabel("Wyniki benchmarku: Niedostępne")
        layout.addWidget(self.benchmark_results_label)

        if self.user_role == "admin":
            print("[DEBUG] Użytkownik jest adminem, dodaję elementy interfejsu dla admina")
            # Wybór modelu
            self.model_combo = QComboBox()
            layout.addWidget(QLabel("Wybierz model:"))
            layout.addWidget(self.model_combo)
            self.load_models_for_benchmark()
            
            # Etykieta wybranego folderu
            self.selected_folder_label = QLabel("Wybrany folder: Brak")
            layout.addWidget(self.selected_folder_label)
            
            # Przycisk wyboru folderu
            self.select_folder_btn = QPushButton("Wybierz folder z danymi")
            self.select_folder_btn.clicked.connect(self.select_folder)
            layout.addWidget(self.select_folder_btn)
            
            # Przycisk uruchamiania benchmarku
            self.run_benchmark_btn = QPushButton("Uruchom Benchmark")
            self.run_benchmark_btn.clicked.connect(self.run_benchmark)
            layout.addWidget(self.run_benchmark_btn)
        else:
            print("[DEBUG] Użytkownik nie jest adminem, pomijam elementy interfejsu dla admina")

        self.benchmark_tab.setLayout(layout)

    def select_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z obrazami i annotacjami")
        if folder_path:
            self.selected_folder = folder_path
            self.selected_folder_label.setText(f"Wybrany folder: {self.selected_folder}")
            print(f"[DEBUG] Wybrano folder: {self.selected_folder}")
        else:
            self.selected_folder = None
            self.selected_folder_label.setText("Wybrany folder: Brak")
            print("[DEBUG] Nie wybrano folderu")

    def load_models_for_benchmark(self):
        print("[DEBUG] Ładowanie modeli do QComboBox")
        try:
            response = requests.get("http://localhost:8000/models")
            if response.status_code == 200:
                models = response.json()
                print(f"[DEBUG] Załadowano modele: {models}")
                if not models:
                    QtWidgets.QMessageBox.warning(self, "Brak modeli", "Nie znaleziono żadnych modeli w bazie danych.")
                    self.model_combo.addItem("Brak modeli", None)
                    return
                self.model_combo.clear()
                for model in models:
                    display_text = f"{model['algorithm']} - v{model['version']}"
                    # Przechowujemy algorithm, version i name
                    self.model_combo.addItem(display_text, (model['algorithm'], model['version'], model['name']))
            else:
                print(f"[DEBUG] Błąd podczas pobierania modeli: {response.status_code} - {response.text}")
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas pobierania modeli: {response.text}")
        except Exception as e:
            print(f"[DEBUG] Wyjątek podczas ładowania modeli: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas ładowania modeli: {e}")

    def run_benchmark(self):
        if self.user_role != "admin":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Tylko admin może uruchomić benchmark!")
            return

        if not self.selected_folder:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano folderu z danymi!")
            return

        # Pobierz wybrany model (algorithm, version, name)
        selected_model = self.model_combo.currentData()
        if not selected_model:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano modelu!")
            return

        algorithm, model_version, model_name = selected_model

        # Przygotuj pliki do przesłania
        files = []
        for file_name in os.listdir(self.selected_folder):
            file_path = os.path.join(self.selected_folder, file_name)
            if file_name.endswith(('.jpg', '.png')):
                files.append(('images', (file_name, open(file_path, 'rb'), 'image/jpeg')))
            elif file_name.endswith('.json'):
                files.append(('annotations', (file_name, open(file_path, 'rb'), 'application/json')))

        if not any(f[0] == 'images' for f in files):
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wybrany folder nie zawiera obrazów (.jpg, .png)!")
            return

        if not any(f[0] == 'annotations' for f in files):
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wybrany folder nie zawiera annotacji (.json)!")
            return

        print(f"[DEBUG] Przesyłane pliki: {[f[1][0] for f in files if f[0] == 'images']}")

        # Najpierw przygotuj dane w kontenerze
        try:
            headers = {"X-User-Role": self.user_role}
            print(f"[DEBUG] Wysyłanie żądania do http://localhost:8000/prepare_benchmark_data z nagłówkami: {headers}")
            response = requests.post("http://localhost:8000/prepare_benchmark_data", files=files, headers=headers)
            print(f"[DEBUG] Odpowiedź serwera (prepare_benchmark_data): status={response.status_code}, treść={response.text}")
            if response.status_code != 200:
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas przygotowywania danych: {response.text}")
                return
        except Exception as e:
            print(f"[DEBUG] Wyjątek podczas przygotowywania danych: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas przygotowywania danych: {e}")
            return
        finally:
            # Zamknij wszystkie otwarte pliki
            for _, (filename, file, _) in files:
                file.close()

        # Ścieżki w kontenerze
        image_folder = "/app/backend/data/test/images"
        annotation_path = "/app/backend/data/test/annotations"

        # Przygotuj dane do żądania benchmarku
        data = {
            "algorithm": algorithm,
            "model_version": model_version,
            "model_name": model_name,  # Przekazujemy name zamiast version do identyfikacji
            "image_folder": image_folder,
            "annotation_path": annotation_path
        }

        try:
            headers = {"X-User-Role": self.user_role}
            print(f"[DEBUG] Wysyłanie żądania do http://localhost:8000/benchmark z nagłówkami: {headers}")
            response = requests.post("http://localhost:8000/benchmark", json=data, headers=headers)
            print(f"[DEBUG] Odpowiedź serwera: status={response.status_code}, treść={response.text}")
            if response.status_code == 200:
                print("[DEBUG] Benchmark zakończony sukcesem")
                self.update_benchmark_results()
                QtWidgets.QMessageBox.information(self, "Sukces", "Benchmark uruchomiony pomyślnie!")
            else:
                print(f"[DEBUG] Błąd podczas benchmarku: {response.status_code} - {response.text}")
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas uruchamiania benchmarku: {response.text}")
        except Exception as e:
            print(f"[DEBUG] Wyjątek podczas benchmarku: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd: {e}")

    def update_benchmark_results(self):
        print("[DEBUG] Aktualizacja wyników benchmarku")
        try:
            headers = {"X-User-Role": self.user_role}
            response = requests.get("http://localhost:8000/get_benchmark_results", headers=headers)
            print(f"[DEBUG] Odpowiedź serwera (get_benchmark_results): status={response.status_code}, treść={response.text}")
            if response.status_code == 200:
                results = response.json()
                print(f"[DEBUG] Wyniki benchmarku: {results}")
                self.benchmark_results_label.setText(
                    f"Wyniki benchmarku: MAE = {results.get('MAE', 'N/A')} "
                    f"(Model: {results.get('algorithm', 'N/A')} v{results.get('model_version', 'N/A')})"
                )
            else:
                print(f"[DEBUG] Brak wyników benchmarku: {response.status_code} - {response.text}")
                self.benchmark_results_label.setText("Wyniki benchmarku: Niedostępne")
        except Exception as e:
            print(f"[DEBUG] Wyjątek podczas aktualizacji wyników: {e}")
            self.benchmark_results_label.setText(f"Wyniki benchmarku: Błąd - {e}")