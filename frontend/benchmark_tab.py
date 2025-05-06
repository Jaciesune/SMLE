import requests
import os
import json
import logging
from PyQt5 import QtWidgets

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BenchmarkTab(QtWidgets.QWidget):
    def __init__(self, user_role):
        super().__init__()
        self.user_role = user_role
        self.api_url = "http://localhost:8000"
        self.selected_folder = None
        self.init_ui()

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout()

        # Panel sterowania (tylko dla admina)
        if self.user_role == "admin":
            # Wybór folderu
            folder_layout = QtWidgets.QHBoxLayout()
            self.folder_label = QtWidgets.QLabel("Brak wybranego folderu")
            folder_button = QtWidgets.QPushButton("Wybierz folder")
            folder_button.clicked.connect(self.select_folder)
            folder_layout.addWidget(self.folder_label)
            folder_layout.addWidget(folder_button)
            self.layout.addLayout(folder_layout)

            # Wybór modelu
            model_layout = QtWidgets.QHBoxLayout()
            model_label = QtWidgets.QLabel("Wybierz model:")
            self.model_combo = QtWidgets.QComboBox()
            self.load_models()
            model_layout.addWidget(model_label)
            model_layout.addWidget(self.model_combo)
            self.layout.addLayout(model_layout)

            # Przycisk do uruchomienia benchmarku
            run_button = QtWidgets.QPushButton("Uruchom Benchmark")
            run_button.clicked.connect(self.run_benchmark)
            self.layout.addWidget(run_button)

            # Przycisk do porównania modeli
            compare_button = QtWidgets.QPushButton("Porównaj modele")
            compare_button.clicked.connect(self.compare_models)
            self.layout.addWidget(compare_button)

        # Wyniki benchmarku (dostępne dla wszystkich)
        self.benchmark_results = QtWidgets.QTextEdit()
        self.benchmark_results.setReadOnly(True)
        self.layout.addWidget(self.benchmark_results)

        self.setLayout(self.layout)
        self.update_benchmark_results()

    def select_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz folder z obrazami i annotacjami")
        if folder_path:
            self.selected_folder = folder_path
            self.folder_label.setText(f"Wybrany folder: {self.selected_folder}")
            logger.debug(f"[DEBUG] Wybrano folder: {self.selected_folder}")
        else:
            self.selected_folder = None
            self.folder_label.setText("Wybrany folder: Brak")
            logger.debug("[DEBUG] Nie wybrano folderu")

    def load_models(self):
        logger.debug("[DEBUG] Ładowanie modeli do QComboBox")
        try:
            response = requests.get(f"{self.api_url}/models")
            if response.status_code == 200:
                models = response.json()
                logger.debug(f"[DEBUG] Załadowano modele: {models}")
                if not models:
                    QtWidgets.QMessageBox.warning(self, "Brak modeli", "Nie znaleziono żadnych modeli w bazie danych.")
                    self.model_combo.addItem("Brak modeli", None)
                    return
                self.model_combo.clear()
                for model in models:
                    display_text = model['display_name']
                    self.model_combo.addItem(display_text, model)
            else:
                logger.debug(f"[DEBUG] Błąd podczas pobierania modeli: {response.status_code} - {response.text}")
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas pobierania modeli: {response.text}")
        except Exception as e:
            logger.debug(f"[DEBUG] Wyjątek podczas ładowania modeli: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas ładowania modeli: {e}")

    def run_benchmark(self):
        if not self.selected_folder:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano folderu z danymi!")
            return

        # Pobierz wybrany model
        selected_model = self.model_combo.currentData()
        if not selected_model:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano modelu!")
            return

        # Przygotuj pliki do przesłania
        files = []
        image_files = []
        annotation_files = []
        for file_name in os.listdir(self.selected_folder):
            file_path = os.path.join(self.selected_folder, file_name)
            if file_name.lower().endswith(('.jpg', '.png')):
                image_files.append(('images', (file_name, open(file_path, 'rb'), 'image/jpeg')))
            elif file_name.lower().endswith('.json'):
                annotation_files.append(('annotations', (file_name, open(file_path, 'rb'), 'application/json')))

        if not image_files:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wybrany folder nie zawiera obrazów (.jpg, .png)!")
            return

        if not annotation_files:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wybrany folder nie zawiera annotacji (.json)!")
            return

        files.extend(image_files)
        files.extend(annotation_files)

        logger.debug(f"[DEBUG] Przesyłane obrazy: {[f[1][0] for f in image_files]}")
        logger.debug(f"[DEBUG] Przesyłane annotacje: {[f[1][0] for f in annotation_files]}")

        # Przygotuj dane JSON
        data = {
            "algorithm": selected_model['algorithm'],
            "model_version": selected_model['version'],
            "model_name": selected_model['name'],
            "image_folder": "",
            "source_folder": self.selected_folder,
            "annotation_path": ""
        }
        json_data = json.dumps(data)

        # Połącz dane JSON i pliki w jednym żądaniu
        try:
            headers = {"X-User-Role": self.user_role}
            files.append(('json_data', (None, json_data, 'application/json')))
            logger.debug(f"[DEBUG] Wysyłanie żądania do {self.api_url}/run_benchmark z nagłówkami: {headers}, pliki={len(files)}")
            response = requests.post(f"{self.api_url}/run_benchmark", files=files, headers=headers)
            logger.debug(f"[DEBUG] Odpowiedź serwera: status={response.status_code}, treść={response.text}")
            if response.status_code == 200:
                logger.debug("[DEBUG] Benchmark zakończony sukcesem")
                self.update_benchmark_results()
                QtWidgets.QMessageBox.information(self, "Sukces", "Benchmark uruchomiony pomyślnie!")
            else:
                logger.debug(f"[DEBUG] Błąd podczas benchmarku: {response.status_code} - {response.text}")
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas uruchamiania benchmarku: {response.text}")
        except Exception as e:
            logger.debug(f"[DEBUG] Wyjątek podczas benchmarku: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd: {e}")
        finally:
            # Zamknij wszystkie otwarte pliki
            for _, (filename, file, _) in files[:-1]:  # Pomijamy pole 'json_data', które nie jest plikiem
                file.close()
                logger.debug(f"[DEBUG] Zamknięto plik: {filename}")

    def update_benchmark_results(self):
        try:
            headers = {"X-User-Role": self.user_role}
            response = requests.get(f"{self.api_url}/get_benchmark_results", headers=headers)
            logger.debug(f"[DEBUG] Odpowiedź z /get_benchmark_results: status={response.status_code}, treść={response.text}")
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
                result_text += f"Folder źródłowy: {results.get('source_folder', 'Brak danych')}\n"
                result_text += f"Czas: {results.get('timestamp', 'Brak danych')}\n"
                result_text += "-" * 40 + "\n"

            self.benchmark_results.setText(result_text)
        except Exception as e:
            logger.debug(f"[DEBUG] Błąd podczas aktualizacji wyników benchmarku: {e}")
            self.benchmark_results.setText("Brak wyników benchmarku.")

    def compare_models(self):
        try:
            response = requests.get(f"{self.api_url}/compare_models")
            logger.debug(f"[DEBUG] Odpowiedź z /compare_models: status={response.status_code}, treść={response.text}")
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
            comparison_text = "Porównanie modeli według danych wejściowych:\n\n"
            for dataset_result in results:
                dataset = dataset_result["dataset"]
                comparison_text += f"📁 Zestaw danych: {dataset}\n"
                comparison_text += "Wyniki modeli:\n"
                for model_result in dataset_result["results"]:
                    comparison_text += (
                        f"  - Model: {model_result['model']}, "
                        f"Nazwa: {model_result['model_name']}, "
                        f"Skuteczność: {model_result['effectiveness']}%, "
                        f"MAE: {model_result['mae']}, "
                        f"Czas: {model_result['timestamp']}\n"
                    )
                best_model = dataset_result["best_model"]
                comparison_text += (
                    f"🏆 Najlepszy model dla tego zestawu:\n"
                    f"    Model: {best_model['model']}, "
                    f"Nazwa: {best_model['model_name']}, "
                    f"Skuteczność: {best_model['effectiveness']}%\n\n"
                )

            if best_model_info:
                comparison_text += (
                    f"🌟 Najlepszy model ogólny:\n"
                    f"Zestaw danych: {best_model_info['dataset']}\n"
                    f"Model: {best_model_info['model']}\n"
                    f"Nazwa: {best_model_info['model_name']}\n"
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
            logger.debug(f"[DEBUG] Wyjątek podczas porównania modeli: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas porównania modeli: {e}")