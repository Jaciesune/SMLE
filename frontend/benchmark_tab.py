import requests
import os
import json
import logging
import winsound
from PyQt5 import QtWidgets, QtCore

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LoadingDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setObjectName("loading_dialog")
        self.setFixedSize(200, 100)
        self.setStyleSheet("""
            QDialog#loading_dialog {
                background-color: #222831;
                border: 1px solid #948979;
                border-radius: 12px;
                min-width: 300px;
                max-width: none;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 18px;
                font-weight: bold;
            }
        """)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.dots_label = QtWidgets.QLabel("Uruchamianie benchmarku" + ".")
        self.dots_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.dots_label)
        self.setLayout(layout)

        # Timer do animacji kropek
        self.dot_count = 1
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_dots)
        self.timer.start(500)  # Zmiana co 500 ms

    def update_dots(self):
        self.dot_count = (self.dot_count % 3) + 1
        self.dots_label.setText("Uruchamianie benchmarku" + "." * self.dot_count)

    def stop(self):
        self.timer.stop()

class ProgressThread(QtCore.QThread):
    stop_progress = QtCore.pyqtSignal()

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.running = True

    def run(self):
        while self.running:
            self.msleep(100)

    def stop(self):
        self.running = False
        self.dialog.stop()
        self.stop_progress.emit()

class BenchmarkThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, api_url, selected_folder, selected_model, user_role):
        super().__init__()
        self.api_url = api_url
        self.selected_folder = selected_folder
        self.selected_model = selected_model
        self.user_role = user_role
        self.files = []

    def run(self):
        try:
            # Przygotuj pliki do przesłania
            image_files = []
            annotation_files = []
            for file_name in os.listdir(self.selected_folder):
                file_path = os.path.join(self.selected_folder, file_name)
                if file_name.lower().endswith(('.jpg', '.png')):
                    image_files.append(('images', (file_name, open(file_path, 'rb'), 'image/jpeg')))
                elif file_name.lower().endswith('.json'):
                    annotation_files.append(('annotations', (file_name, open(file_path, 'rb'), 'application/json')))

            if not image_files:
                raise ValueError("Wybrany folder nie zawiera obrazów (.jpg, .png)!")
            if not annotation_files:
                raise ValueError("Wybrany folder nie zawiera annotacji (.json)!")

            self.files.extend(image_files)
            self.files.extend(annotation_files)

            logger.debug(f"[DEBUG] Przesyłane obrazy: {[f[1][0] for f in image_files]}")
            logger.debug(f"[DEBUG] Przesyłane annotacje: {[f[1][0] for f in annotation_files]}")

            # Przygotuj dane JSON
            data = {
                "algorithm": self.selected_model['algorithm'],
                "model_version": self.selected_model['version'],
                "model_name": self.selected_model['name'],
                "image_folder": "",
                "source_folder": self.selected_folder,
                "annotation_path": ""
            }
            json_data = json.dumps(data)

            # Połącz dane JSON i pliki w jednym żądaniu
            headers = {"X-User-Role": self.user_role}
            self.files.append(('json_data', (None, json_data, 'application/json')))
            logger.debug(f"[DEBUG] Wysyłanie żądania do {self.api_url}/run_benchmark z nagłówkami: {headers}, pliki={len(self.files)}")
            response = requests.post(f"{self.api_url}/run_benchmark", files=self.files, headers=headers)
            logger.debug(f"[DEBUG] Odpowiedź serwera: status={response.status_code}, treść={response.text}")

            if response.status_code == 200:
                logger.debug("[DEBUG] Benchmark zakończony sukcesem")
                self.finished.emit({"status": "success", "response": response.text})
            else:
                raise ValueError(f"Błąd podczas benchmarku: {response.status_code} - {response.text}")

        except Exception as e:
            logger.debug(f"[DEBUG] Wyjątek podczas benchmarku: {e}")
            self.error.emit(str(e))
        finally:
            # Zamknij wszystkie otwarte pliki
            for _, (filename, file, _) in self.files[:-1]:  # Pomijamy pole 'json_data'
                file.close()
                logger.debug(f"[DEBUG] Zamknięto plik: {filename}")

class BenchmarkTab(QtWidgets.QWidget):
    def __init__(self, user_role, api_url):
        super().__init__()
        self.user_role = user_role
        self.api_url = api_url
        self.selected_folder = None
        self.benchmark_thread = None
        self.progress_thread = None
        self.loading_dialog = None
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

        if self.benchmark_thread and self.benchmark_thread.isRunning():
            logger.warning("Benchmark już w toku, zignorowano ponowne kliknięcie.")
            return

        # Pokazanie okna z animacją kropek
        self.loading_dialog = LoadingDialog(self)
        self.loading_dialog.move(
            self.width() // 2 - self.loading_dialog.width() // 2,
            self.height() // 2 - self.loading_dialog.height() // 2
        )
        self.loading_dialog.show()
        QtWidgets.QApplication.processEvents()

        self.progress_thread = ProgressThread(self.loading_dialog)
        self.progress_thread.start()

        self.benchmark_thread = BenchmarkThread(
            self.api_url, self.selected_folder, selected_model, self.user_role
        )
        self.benchmark_thread.finished.connect(self.on_benchmark_finished)
        self.benchmark_thread.error.connect(self.on_benchmark_error)
        self.benchmark_thread.start()

    def on_benchmark_finished(self, result):
        logger.debug("Benchmark zakończony")
        winsound.PlaySound("SystemNotification", winsound.SND_ALIAS)
        self.loading_dialog.close()

        if self.progress_thread:
            self.progress_thread.stop()
            self.progress_thread.wait()

        self.update_benchmark_results()
        QtWidgets.QMessageBox.information(self, "Sukces", "Benchmark uruchomiony pomyślnie!")

    def on_benchmark_error(self, error_message):
        logger.error(f"Błąd podczas benchmarku: {error_message}")
        self.loading_dialog.close()

        if self.progress_thread:
            self.progress_thread.stop()
            self.progress_thread.wait()

        QtWidgets.QMessageBox.critical(self, "Błąd", f"Błąd podczas uruchamiania benchmarku: {error_message}")

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