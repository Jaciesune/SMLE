import json
import subprocess
from PyQt5 import QtWidgets
from fastapi import logger
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
        
        # Tworzymy zakładkę Benchmark
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
        layout = QtWidgets.QVBoxLayout()

        # Wybór folderu
        folder_layout = QtWidgets.QHBoxLayout()
        self.folder_label = QtWidgets.QLabel("Brak wybranego folderu")
        folder_button = QtWidgets.QPushButton("Wybierz folder")
        folder_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(folder_button)
        layout.addLayout(folder_layout)

        # Wybór modelu
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Wybierz model:")
        self.model_combo = QtWidgets.QComboBox()
        self.load_models_for_benchmark()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Przycisk do uruchomienia benchmarku
        run_button = QtWidgets.QPushButton("Uruchom Benchmark")
        run_button.clicked.connect(self.run_benchmark)
        layout.addWidget(run_button)

        # Przycisk do porównania modeli
        compare_button = QtWidgets.QPushButton("Porównaj modele")
        compare_button.clicked.connect(self.compare_models)
        layout.addWidget(compare_button)

        # Wyniki benchmarku
        self.benchmark_results = QtWidgets.QTextEdit()
        self.benchmark_results.setReadOnly(True)
        layout.addWidget(self.benchmark_results)

        # Ustawienie layoutu dla zakładki
        self.benchmark_tab.setLayout(layout)

    def select_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z obrazami i annotacjami")
        if folder_path:
            self.selected_folder = folder_path
            self.folder_label.setText(f"Wybrany folder: {self.selected_folder}")
            print(f"[DEBUG] Wybrano folder: {self.selected_folder}")
        else:
            self.selected_folder = None
            self.folder_label.setText("Wybrany folder: Brak")
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
                    display_text = model['display_name']  # Używamy display_name z API
                    # Przechowujemy cały słownik modelu
                    self.model_combo.addItem(display_text, model)
            else:
                print(f"[DEBUG] Błąd podczas pobierania modeli: {response.status_code} - {response.text}")
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas pobierania modeli: {response.text}")
        except Exception as e:
            print(f"[DEBUG] Wyjątek podczas ładowania modeli: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas ładowania modeli: {e}")

    def analyze_with_model(self, image_path, algorithm, model_file):
        logger.debug(f"[DEBUG] Rozpoczynanie analizy: image_path={image_path}, algorithm={algorithm}, model_file={model_file}")
        try:
            if algorithm == "Mask R-CNN":
                script_path = "/app/backend/Mask_RCNN/scripts/detect.py"
                cmd = ["python", script_path, "--image", image_path, "--model", model_file]
            elif algorithm == "MCNN":
                script_path = "/app/backend/MCNN/scripts/detect.py"
                cmd = ["python", script_path, "--image", image_path, "--model", model_file]
            elif algorithm == "FasterRCNN":
                script_path = "/app/backend/FasterRCNN/scripts/detect.py"
                cmd = ["python", script_path, "--image", image_path, "--model", model_file]
            else:
                logger.error(f"[DEBUG] Nieobsługiwany algorytm: {algorithm}")
                return f"Błąd: Nieobsługiwany algorytm: {algorithm}", 0

            logger.debug(f"[DEBUG] Wykonywanie komendy: {cmd}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.debug(f"[DEBUG] Wynik komendy: stdout={result.stdout}, stderr={result.stderr}")

            if result.returncode != 0:
                logger.error(f"[DEBUG] Błąd detekcji: {result.stderr}")
                return f"Błąd: {result.stderr}", 0

            # Przyjmijmy, że skrypt zwraca liczbę wykrytych obiektów w stdout
            try:
                num_predicted = int(result.stdout.strip())
            except ValueError:
                logger.error(f"[DEBUG] Nieprawidłowy format wyniku: {result.stdout}")
                return f"Błąd: Nieprawidłowy format wyniku: {result.stdout}", 0

            logger.debug(f"[DEBUG] Liczba wykrytych obiektów: {num_predicted}")
            return "Sukces", num_predicted
        except Exception as e:
            logger.error(f"[DEBUG] Wyjątek w analyze_with_model: {str(e)}")
            return f"Błąd: {str(e)}", 0

    def run_benchmark(self):
        if self.user_role != "admin":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Tylko admin może uruchomić benchmark!")
            return

        if not self.selected_folder:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano folderu z danymi!")
            return

        # Pobierz wybrany model (cały słownik modelu)
        selected_model = self.model_combo.currentData()
        if not selected_model:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano modelu!")
            return

        # Pobierz wartości ze słownika
        algorithm = selected_model['algorithm']
        model_version = selected_model['version']
        model_name = selected_model['name']

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
        try:
            headers = {"X-User-Role": self.user_role}
            response = requests.get("http://localhost:8000/get_benchmark_results", headers=headers)
            print(f"[DEBUG] Odpowiedź z /get_benchmark_results: status={response.status_code}, treść={response.text}")
            if response.status_code != 200:
                self.benchmark_results.setText("Brak wyników benchmarku lub błąd: " + response.text)
                return

            data = response.json()
            history = data.get("history", [])
            
            if not history:
                self.benchmark_results.setText("Brak wyników benchmarku.")
                return

            result_text = "Historia benchmarków:\n\n"
            for idx, results in enumerate(history, 1):
                result_text += f"Benchmark #{idx}:\n"
                result_text += f"Algorytm: {results.get('algorithm', 'Brak danych')}\n"
                result_text += f"Wersja modelu: {results.get('model_version', 'Brak danych')}\n"
                result_text += f"Nazwa modelu: {results.get('model_name', 'Brak danych')}\n"
                result_text += f"MAE: {results.get('MAE', 'Brak danych')}\n"
                result_text += f"Skuteczność: {results.get('effectiveness', 'Brak danych')}%\n"
                result_text += f"Czas: {results.get('timestamp', 'Brak danych')}\n"
                result_text += "-" * 40 + "\n"

            self.benchmark_results.setText(result_text)
        except Exception as e:
            print(f"[DEBUG] Błąd podczas aktualizacji wyników benchmarku: {e}")
            self.benchmark_results.setText("Brak wyników benchmarku.")

    def compare_models(self):
        try:
            response = requests.get("http://localhost:8000/compare_models")
            print(f"[DEBUG] Odpowiedź z /compare_models: status={response.status_code}, treść={response.text}")
            if response.status_code != 200:
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas pobierania porównania modeli: {response.text}")
                return

            comparison_data = response.json()
            results = comparison_data.get("results", [])
            best_model_info = comparison_data.get("best_model", None)

            if not results:
                QtWidgets.QMessageBox.information(self, "Informacja", "Brak wyników benchmarków do porównania.")
                return

            # Tworzenie tekstu do wyświetlenia
            comparison_text = "Porównanie modeli:\n\n"
            for dataset_result in results:
                dataset = dataset_result["dataset"]
                comparison_text += f"Zbiór danych: {dataset}\n"
                for model_result in dataset_result["results"]:
                    comparison_text += (
                        f"  Model: {model_result['model']}, "
                        f"Nazwa modelu: {model_result['model_name']}, "
                        f"Skuteczność: {model_result['effectiveness']}%, "
                        f"MAE: {model_result['mae']}, "
                        f"Czas: {model_result['timestamp']}\n"
                    )
                best_model = dataset_result["best_model"]
                comparison_text += (
                    f"  Najlepszy model dla tego zbioru: {best_model['model_name']}, "
                    f"Skuteczność: {best_model['effectiveness']}%\n\n"
                )

            if best_model_info:
                comparison_text += (
                    f"Najlepszy model ogólny:\n"
                    f"Zbiór danych: {best_model_info['dataset']}\n"
                    f"Model: {best_model_info['model']}\n"
                    f"Model_name: {best_model_info['model_name']}\n"
                    f"Skuteczność: {best_model_info['effectiveness']}%\n"
                )

            # Wyświetl wyniki w oknie dialogowym
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Porównanie modeli")
            layout = QtWidgets.QVBoxLayout(dialog)

            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setText(comparison_text)
            layout.addWidget(text_edit)

            close_button = QtWidgets.QPushButton("Zamknij")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)

            dialog.setLayout(layout)
            dialog.resize(600, 400)
            dialog.exec_()

        except Exception as e:
            print(f"[DEBUG] Wyjątek podczas porównania modeli: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas porównania modeli: {e}")