from PyQt5 import QtWidgets
import requests
import os
import uuid
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DatasetCreationTab(QtWidgets.QWidget):
    def __init__(self, user_role):
        super().__init__()
        self.user_role = user_role
        self.api_url = "http://localhost:8000"  # Zmień na "http://host.docker.internal:8000" jeśli WSL2
        self.job_name = None
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        self.input_dir_label = QtWidgets.QLabel("Katalog z obrazami i adnotacjami (.jpg i .json):")
        layout.addWidget(self.input_dir_label)

        self.input_dir_edit = QtWidgets.QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        layout.addWidget(self.input_dir_edit)

        self.input_dir_button = QtWidgets.QPushButton("Wybierz katalog")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        layout.addWidget(self.input_dir_button)

        self.create_button = QtWidgets.QPushButton("Utwórz dataset")
        self.create_button.clicked.connect(self.create_dataset)
        layout.addWidget(self.create_button)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

    def select_input_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz katalog z obrazami i adnotacjami")
        if directory:
            self.input_dir_edit.setText(directory)
            self.log_text.append(f"Wybrano katalog: {directory}")

    def create_dataset(self):
        input_dir = self.input_dir_edit.text()
        if not input_dir:
            self.log_text.append("Proszę wybrać katalog z danymi!")
            return

        self.job_name = f"dataset_{uuid.uuid4().hex}"
        self.log_text.append(f"Nazwa zadania: {self.job_name}")

        json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
        if not json_files or not image_files:
            self.log_text.append(f"Błąd: Brak plików .json lub .jpg w katalogu {input_dir}.")
            return

        try:
            logger.debug("Wysyłam żądanie do %s/create_dataset: job_name=%s, %d obrazów, %d adnotacji",
                         self.api_url, self.job_name, len(image_files), len(json_files))
            files = []
            for fname in json_files + image_files:
                fpath = os.path.join(input_dir, fname)
                mime_type = "image/jpeg" if fname.endswith(".jpg") else "application/json"
                files.append(('files', (fname, open(fpath, 'rb'), mime_type)))

            data = {'job_name': self.job_name}
            response = requests.post(
                f"{self.api_url}/create_dataset",
                files=files,
                data=data
            )
            response.raise_for_status()
            self.log_text.append("Tworzenie datasetu zakończone pomyślnie!")

            # Poproś użytkownika o wybranie lokalizacji zapisu
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Zapisz wyniki", f"{self.job_name}_results.zip", "ZIP files (*.zip)"
            )
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                self.log_text.append(f"Pobrano wyniki do: {save_path}")
            else:
                self.log_text.append("Anulowano zapisywanie wyników.")
        except requests.exceptions.HTTPError as e:
            logger.error("Błąd HTTP podczas tworzenia datasetu: %s", e)
            self.log_text.append(f"Błąd podczas tworzenia datasetu: {e}")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas tworzenia datasetu: %s", e)
            self.log_text.append(f"Błąd podczas tworzenia datasetu: {e}")
        finally:
            for _, file_tuple in files:
                file_tuple[1].close()