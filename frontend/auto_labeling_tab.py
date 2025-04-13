from PyQt5 import QtWidgets
import requests
import os
import re
import uuid
import time
import shutil
import logging
import glob

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AutoLabelingTab(QtWidgets.QWidget):
    def __init__(self, user_role):
        super().__init__()
        self.user_role = user_role
        self.api_url = "http://localhost:8000"  # Zmień na "http://host.docker.internal:8000" jeśli WSL2
        self.job_name = None
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        self.input_dir_label = QtWidgets.QLabel("Katalog ze zdjęciami:")
        layout.addWidget(self.input_dir_label)

        self.input_dir_edit = QtWidgets.QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        layout.addWidget(self.input_dir_edit)

        self.input_dir_button = QtWidgets.QPushButton("Wybierz katalog")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        layout.addWidget(self.input_dir_button)

        self.model_version_label = QtWidgets.QLabel("Wybierz model (tylko Mask R-CNN):")
        layout.addWidget(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        # Nie wywołujemy update_model_versions tutaj, zrobimy to później
        layout.addWidget(self.model_version_combo)

        self.label_button = QtWidgets.QPushButton("Uruchom automatyczne labelowanie")
        self.label_button.clicked.connect(self.run_auto_labeling)
        layout.addWidget(self.label_button)

        self.download_button = QtWidgets.QPushButton("Pobierz wyniki")
        self.download_button.clicked.connect(self.download_results)
        self.download_button.setEnabled(False)
        layout.addWidget(self.download_button)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Teraz, gdy self.log_text jest zainicjalizowany, możemy wywołać update_model_versions
        self.update_model_versions()

        self.setLayout(layout)

    def select_input_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz katalog ze zdjęciami")
        if directory:
            self.input_dir_edit.setText(directory)
            self.log_text.append(f"Wybrano katalog: {directory}")

    def update_model_versions(self):
        try:
            logger.debug("Pobieram listę modeli z %s/model_versions", self.api_url)
            response = requests.get(f"{self.api_url}/model_versions_maskrcnn")
            response.raise_for_status()
            model_versions = response.json()
            self.model_version_combo.clear()
            if model_versions:
                self.model_version_combo.addItems(model_versions)
            else:
                self.model_version_combo.addItem("Brak dostępnych modeli")
                self.log_text.append("Brak dostępnych modeli dla Mask R-CNN.")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas pobierania modeli: %s", e)
            self.log_text.append(f"Błąd podczas pobierania modeli: {e}")
            self.model_version_combo.clear()
            self.model_version_combo.addItem("Brak dostępnych modeli")

    def run_auto_labeling(self):
        input_dir = self.input_dir_edit.text()
        model_version = self.model_version_combo.currentText()

        if not input_dir:
            self.log_text.append("Proszę wybrać katalog ze zdjęciami!")
            return
        if not model_version or model_version == "Brak dostępnych modeli":
            self.log_text.append("Proszę wybrać model!")
            return

        self.job_name = f"auto_label_{uuid.uuid4().hex}"
        self.log_text.append(f"Nazwa zadania: {self.job_name}")

        # Zbierz pliki obrazów
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
        if not image_paths:
            self.log_text.append(f"Błąd: Brak obrazów .jpg w katalogu {input_dir}.")
            return

        try:
            logger.debug("Wysyłam żądanie do %s/auto_label: job_name=%s, model_version=%s, %d obrazów",
                         self.api_url, self.job_name, model_version, len(image_paths))
            files = [('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in image_paths]
            data = {'job_name': self.job_name, 'model_version': model_version}
            response = requests.post(
                f"{self.api_url}/auto_label",
                files=files,
                data=data
            )
            response.raise_for_status()
            self.log_text.append("Labelowanie zakończone pomyślnie!")
            self.download_button.setEnabled(True)
            zip_path = f"{self.job_name}_results.zip"
            with open(zip_path, "wb") as f:
                f.write(response.content)
            self.log_text.append(f"Pobrano wyniki do: {zip_path}")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas labelowania: %s", e)
            self.log_text.append(f"Błąd podczas labelowania: {e}")
            self.download_button.setEnabled(False)
        finally:
            for _, file_tuple in files:
                file_tuple[1].close()

    def download_results(self):
        if not self.job_name:
            self.log_text.append("Błąd: Brak nazwy zadania. Najpierw uruchom labelowanie.")
            return

        zip_path = f"{self.job_name}_results.zip"
        if not os.path.exists(zip_path):
            self.log_text.append(f"Błąd: Plik {zip_path} nie istnieje.")
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Zapisz wyniki", f"{self.job_name}_results.zip", "ZIP files (*.zip)"
        )
        if save_path:
            shutil.move(zip_path, save_path)
            self.log_text.append(f"Wyniki zapisane do: {save_path}")
        else:
            self.log_text.append("Anulowano zapisywanie wyników.")