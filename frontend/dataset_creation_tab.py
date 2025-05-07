from PyQt5 import QtWidgets, QtCore
import requests
import os
import logging
import random

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCreationTab(QtWidgets.QWidget):
    def __init__(self, user_role, username, api_url):
        super().__init__()
        self.user_role = user_role
        self.username = username
        self.dataset_name = None
        self.train_ratio = 0.6
        self.val_ratio = 0.3
        self.test_ratio = 0.1
        self.api_url = api_url
        self.selected_dataset_images = {"train": [], "val": [], "test": []}
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # Sekcja tworzenia nowego datasetu
        self.new_dataset_layout = QtWidgets.QHBoxLayout()
        self.new_dataset_label = QtWidgets.QLabel("Nazwa nowego datasetu:")
        self.new_dataset_layout.addWidget(self.new_dataset_label)
        self.new_dataset_input = QtWidgets.QLineEdit()
        self.new_dataset_input.setPlaceholderText("Wpisz nazwę datasetu")
        self.new_dataset_layout.addWidget(self.new_dataset_input)
        
        self.train_ratio_label = QtWidgets.QLabel("Train (%):")
        self.new_dataset_layout.addWidget(self.train_ratio_label)
        self.train_ratio_input = QtWidgets.QLineEdit("60")
        self.train_ratio_input.setFixedWidth(50)
        self.new_dataset_layout.addWidget(self.train_ratio_input)
        
        self.val_ratio_label = QtWidgets.QLabel("Val (%):")
        self.new_dataset_layout.addWidget(self.val_ratio_label)
        self.val_ratio_input = QtWidgets.QLineEdit("30")
        self.val_ratio_input.setFixedWidth(50)
        self.new_dataset_layout.addWidget(self.val_ratio_input)
        
        self.test_ratio_label = QtWidgets.QLabel("Test (%):")
        self.new_dataset_layout.addWidget(self.test_ratio_label)
        self.test_ratio_input = QtWidgets.QLineEdit("10")
        self.test_ratio_input.setFixedWidth(50)
        self.new_dataset_layout.addWidget(self.test_ratio_input)
        
        self.create_dataset_button = QtWidgets.QPushButton("Utwórz nowy dataset")
        self.create_dataset_button.clicked.connect(self.create_new_dataset)
        self.new_dataset_layout.addWidget(self.create_dataset_button)
        layout.addLayout(self.new_dataset_layout)

        # Lista rozwijana datasetów
        self.dataset_list_label = QtWidgets.QLabel("Wybierz dataset:")
        layout.addWidget(self.dataset_list_label)
        self.dataset_list = QtWidgets.QComboBox()
        self.dataset_list.addItem("Wybierz dataset")
        self.dataset_list.currentTextChanged.connect(self.load_dataset)
        layout.addWidget(self.dataset_list)

        # Przyciski zwiększania i usuwania datasetu
        self.dataset_actions_layout = QtWidgets.QHBoxLayout()
        self.increase_button = QtWidgets.QPushButton("Zwiększ dataset")
        self.increase_button.clicked.connect(self.increase_dataset)
        self.increase_button.setEnabled(False)
        self.dataset_actions_layout.addWidget(self.increase_button)
        
        self.delete_button = QtWidgets.QPushButton("Usuń dataset")
        self.delete_button.clicked.connect(self.delete_dataset)
        self.delete_button.setEnabled(False)
        self.dataset_actions_layout.addWidget(self.delete_button)
        layout.addLayout(self.dataset_actions_layout)

        # Przyciski pobierania nad kolumnami
        self.download_buttons_layout = QtWidgets.QHBoxLayout()
        for subset in ["train", "val", "test"]:
            button = QtWidgets.QPushButton(f"Pobierz {subset.capitalize()}")
            button.clicked.connect(lambda _, s=subset: self.download_subset(s))
            button.setEnabled(False)
            self.download_buttons_layout.addWidget(button)
        layout.addLayout(self.download_buttons_layout)

        # Trzy kolumny (train, val, test) z przewijaniem
        self.columns_layout = QtWidgets.QHBoxLayout()
        wysokosc_kolumn = 800
        
        # Train column
        self.train_widget = QtWidgets.QWidget()
        self.train_column = QtWidgets.QVBoxLayout(self.train_widget)
        self.train_scroll = QtWidgets.QScrollArea()
        self.train_scroll.setWidget(self.train_widget)
        self.train_scroll.setWidgetResizable(True)
        self.train_scroll.setMaximumHeight(wysokosc_kolumn)
        self.columns_layout.addWidget(self.train_scroll)

        # Val column
        self.val_widget = QtWidgets.QWidget()
        self.val_column = QtWidgets.QVBoxLayout(self.val_widget)
        self.val_scroll = QtWidgets.QScrollArea()
        self.val_scroll.setWidget(self.val_widget)
        self.val_scroll.setWidgetResizable(True)
        self.val_scroll.setMaximumHeight(wysokosc_kolumn)
        self.columns_layout.addWidget(self.val_scroll)

        # Test column
        self.test_widget = QtWidgets.QWidget()
        self.test_column = QtWidgets.QVBoxLayout(self.test_widget)
        self.test_scroll = QtWidgets.QScrollArea()
        self.test_scroll.setWidget(self.test_widget)
        self.test_scroll.setWidgetResizable(True)
        self.test_scroll.setMaximumHeight(wysokosc_kolumn)
        self.columns_layout.addWidget(self.test_scroll)

        layout.addLayout(self.columns_layout)

        # Przycisk pobierz cały dataset
        self.download_all_button = QtWidgets.QPushButton("Pobierz cały dataset")
        self.download_all_button.clicked.connect(self.download_all)
        self.download_all_button.setEnabled(False)
        layout.addWidget(self.download_all_button)

        # Logi
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setLayout(layout)
        self.update_dataset_list()

    def validate_ratios(self):
        """Sprawdza, czy proporcje są poprawne i sumują się do 100%."""
        try:
            train_ratio = float(self.train_ratio_input.text())
            val_ratio = float(self.val_ratio_input.text())
            test_ratio = float(self.test_ratio_input.text())
            total = train_ratio + val_ratio + test_ratio
            if abs(total - 100.0) > 0.01:
                return False, "Suma proporcji musi wynosić 100%!"
            if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
                return False, "Proporcje nie mogą być ujemne!"
            return True, (train_ratio / 100, val_ratio / 100, test_ratio / 100)
        except ValueError:
            return False, "Proporcje muszą być liczbami!"

    def update_dataset_list(self):
        try:
            response = requests.get(f"{self.api_url}/list_datasets/{self.username}")
            response.raise_for_status()
            datasets = response.json().get("datasets", [])
            self.dataset_list.clear()
            self.dataset_list.addItem("Wybierz dataset")
            self.dataset_list.addItems(datasets)
            if datasets:
                first_dataset = datasets[0]
                self.dataset_list.setCurrentText(first_dataset)
                self.load_dataset(first_dataset)
            else:
                self.log_text.append("Brak istniejących datasetów. Utwórz nowy dataset.")
        except requests.exceptions.RequestException as e:
            logger.warning("Błąd podczas pobierania listy datasetów: %s", e)
            self.dataset_list.clear()
            self.dataset_list.addItem("Błąd ładowania listy datasetów")
            self.log_text.append(f"Błąd podczas pobierania listy datasetów: {e}")

    def create_new_dataset(self):
        dataset_name = self.new_dataset_input.text().strip()
        if not dataset_name:
            self.log_text.append("Proszę wpisać nazwę datasetu!")
            return

        valid, result = self.validate_ratios()
        if not valid:
            self.log_text.append(result)
            return
        self.train_ratio, self.val_ratio, self.test_ratio = result

        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz katalog z obrazami i adnotacjami")
        if not directory:
            self.log_text.append("Nie wybrano katalogu!")
            return

        files = []
        json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
        image_files = [f for f in os.listdir(directory) if f.endswith(".jpg")]
        paired_files = []
        for img in image_files:
            json_name = img.replace(".jpg", ".json")
            if json_name in json_files:
                paired_files.append(img)
            else:
                self.log_text.append(f"Pomijam {img} - brak odpowiadającego pliku JSON.")

        if not paired_files:
            self.log_text.append("Brak par obraz-JSON do dodania!")
            return

        new_files = []
        for fname in paired_files:
            fpath = os.path.join(directory, fname)
            new_files.append(('files', (fname, open(fpath, 'rb'), "image/jpeg")))
            json_path = os.path.join(directory, fname.replace(".jpg", ".json"))
            new_files.append(('files', (fname.replace(".jpg", ".json"), open(json_path, 'rb'), "application/json")))

        random.shuffle(paired_files)
        total_files = len(paired_files)
        train_count = int(total_files * self.train_ratio)
        val_count = int(total_files * self.val_ratio)
        test_count = total_files - train_count - val_count

        train_files = paired_files[:train_count]
        val_files = paired_files[train_count:train_count + val_count]
        test_files = paired_files[train_count + val_count:]

        for fname in train_files:
            self.log_text.append(f"Dodano {fname} do train")
            self.selected_dataset_images["train"].append(fname)
        for fname in val_files:
            self.log_text.append(f"Dodano {fname} do val")
            self.selected_dataset_images["val"].append(fname)
        for fname in test_files:
            self.log_text.append(f"Dodano {fname} do test")
            self.selected_dataset_images["test"].append(fname)

        data = {
            'username': self.username,
            'job_name': dataset_name,
            'train_ratio': str(self.train_ratio),  # Konwersja na string
            'val_ratio': str(self.val_ratio),      # Konwersja na string
            'test_ratio': str(self.test_ratio)     # Konwersja na string
        }
        logger.debug("Wysyłane dane: %s", data)
        try:
            response = requests.post(
                f"{self.api_url}/create_dataset",
                files=new_files,
                data=data
            )
            response.raise_for_status()
            self.log_text.append(f"Utworzono nowy dataset: {dataset_name}")
            self.new_dataset_input.clear()
            self.update_dataset_list()
            self.dataset_list.setCurrentText(dataset_name)
        except requests.exceptions.HTTPError as e:
            error_detail = response.json().get('detail', str(e)) if response.text else str(e)
            logger.error("Błąd podczas tworzenia datasetu: %s, szczegóły: %s", e, error_detail)
            self.log_text.append(f"Błąd podczas tworzenia datasetu: {e}\nSzczegóły: {error_detail}")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas tworzenia datasetu: %s", e)
            self.log_text.append(f"Błąd podczas tworzenia datasetu: {e}")
        finally:
            for _, file_tuple in new_files:
                file_tuple[1].close()

    def load_dataset(self, dataset_name):
        self.dataset_name = dataset_name if dataset_name != "Wybierz dataset" else None
        self.increase_button.setEnabled(self.dataset_name is not None)
        self.delete_button.setEnabled(self.dataset_name is not None)
        self.download_all_button.setEnabled(self.dataset_name is not None)
        for i in range(self.download_buttons_layout.count()):
            button = self.download_buttons_layout.itemAt(i).widget()
            if button:
                button.setEnabled(self.dataset_name is not None)
        self.clear_columns()

        if self.dataset_name:
            try:
                response = requests.get(f"{self.api_url}/dataset_info/{self.username}/{self.dataset_name}")
                response.raise_for_status()
                info = response.json()
                self.selected_dataset_images = {
                    "train": info.get("train", {}).get("images", []),
                    "val": info.get("val", {}).get("images", []),
                    "test": info.get("test", {}).get("images", [])
                }

                self.update_column("train", info.get("train", {}).get("count", 0), self.selected_dataset_images["train"], self.train_column)
                self.update_column("val", info.get("val", {}).get("count", 0), self.selected_dataset_images["val"], self.val_column)
                self.update_column("test", info.get("test", {}).get("count", 0), self.selected_dataset_images["test"], self.test_column)

            except requests.exceptions.RequestException as e:
                logger.error("Błąd podczas ładowania datasetu: %s", e)
                self.log_text.append(f"Błąd podczas ładowania datasetu: {e}")

    def delete_dataset(self):
        if not self.dataset_name or self.dataset_name == "Wybierz dataset":
            self.log_text.append("Proszę wybrać dataset do usunięcia!")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Potwierdzenie",
            f"Czy na pewno chcesz usunąć dataset {self.dataset_name}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                response = requests.delete(f"{self.api_url}/delete_dataset/{self.username}/{self.dataset_name}")
                response.raise_for_status()
                self.log_text.append(f"Usunięto dataset: {self.dataset_name}")
                self.dataset_name = None
                self.clear_columns()
                self.update_dataset_list()
            except requests.exceptions.RequestException as e:
                logger.error("Błąd podczas usuwania datasetu: %s", e)
                self.log_text.append(f"Błąd podczas usuwania datasetu: {e}")

    def clear_columns(self):
        for layout in [self.train_column, self.val_column, self.test_column]:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

    def update_column(self, subset, count, images, layout):
        layout.addWidget(QtWidgets.QLabel(f"{subset.capitalize()} ({count} obrazów)"))
        if count > 0:
            for img in images:
                layout.addWidget(QtWidgets.QLabel(img))
        layout.addStretch()

    def increase_dataset(self):
        if not self.dataset_name:
            self.log_text.append("Proszę wybrać dataset!")
            return

        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz katalog z obrazami i adnotacjami")
        if not directory:
            return

        files = []
        json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
        image_files = [f for f in os.listdir(directory) if f.endswith(".jpg")]
        paired_files = []
        for img in image_files:
            json_name = img.replace(".jpg", ".json")
            if json_name in json_files:
                paired_files.append(img)
            else:
                self.log_text.append(f"Pomijam {img} - brak odpowiadającego pliku JSON.")

        if not paired_files:
            self.log_text.append("Brak par obraz-JSON do dodania!")
            return

        new_files = []
        for fname in paired_files:
            fpath = os.path.join(directory, fname)
            new_files.append(('files', (fname, open(fpath, 'rb'), "image/jpeg")))
            json_path = os.path.join(directory, fname.replace(".jpg", ".json"))
            new_files.append(('files', (fname.replace(".jpg", ".json"), open(json_path, 'rb'), "application/json")))

        unique_files = []
        for fname in paired_files:
            response = requests.get(f"{self.api_url}/dataset_info/{self.username}/{self.dataset_name}")
            response.raise_for_status()
            info = response.json()
            existing_images = set()
            for subset in ["train", "val", "test"]:
                existing_images.update(info.get(subset, {}).get("images", []))
            if fname not in existing_images:
                unique_files.append(fname)

        if not unique_files:
            self.log_text.append("Wszystkie obrazy są już w datasecie!")
            for _, file_tuple in new_files:
                file_tuple[1].close()
            return

        random.shuffle(unique_files)
        total_files = len(unique_files)
        train_count = int(total_files * self.train_ratio)
        val_count = int(total_files * self.val_ratio)
        test_count = total_files - train_count - val_count

        train_files = unique_files[:train_count]
        val_files = unique_files[train_count:train_count + val_count]
        test_files = unique_files[train_count + val_count:]

        for fname in train_files:
            self.log_text.append(f"Dodano {fname} do train")
            self.selected_dataset_images["train"].append(fname)
        for fname in val_files:
            self.log_text.append(f"Dodano {fname} do val")
            self.selected_dataset_images["val"].append(fname)
        for fname in test_files:
            self.log_text.append(f"Dodano {fname} do test")
            self.selected_dataset_images["test"].append(fname)

        data = {
            'username': self.username,
            'job_name': self.dataset_name,
            'train_ratio': str(self.train_ratio),  # Konwersja na string
            'val_ratio': str(self.val_ratio),      # Konwersja na string
            'test_ratio': str(self.test_ratio)     # Konwersja na string
        }
        logger.debug("Wysyłane dane: %s", data)
        try:
            response = requests.post(
                f"{self.api_url}/create_dataset",
                files=new_files,
                data=data
            )
            response.raise_for_status()
            self.log_text.append("Dataset zaktualizowany pomyślnie!")
            self.load_dataset(self.dataset_name)
        except requests.exceptions.HTTPError as e:
            error_detail = response.json().get('detail', str(e)) if response.text else str(e)
            logger.error("Błąd podczas aktualizacji datasetu: %s, szczegóły: %s", e, error_detail)
            self.log_text.append(f"Błąd podczas aktualizacji datasetu: {e}\nSzczegóły: {error_detail}")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas aktualizacji datasetu: %s", e)
            self.log_text.append(f"Błąd podczas aktualizacji datasetu: {e}")
        finally:
            for _, file_tuple in new_files:
                file_tuple[1].close()

    def download_subset(self, subset):
        if not self.dataset_name:
            self.log_text.append("Proszę wybrać dataset!")
            return

        try:
            response = requests.get(
                f"{self.api_url}/download_dataset/{self.username}/{self.dataset_name}/{subset}",
                stream=True
            )
            response.raise_for_status()
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Zapisz wyniki", f"{self.dataset_name}_{subset}_results.zip", "ZIP files (*.zip)"
            )
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                self.log_text.append(f"Pobrano {subset} do: {save_path}")
                requests.delete(f"{self.api_url}/delete_zip/{self.username}/{self.dataset_name}/{subset}")
            else:
                self.log_text.append("Anulowano zapisywanie wyników.")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas pobierania %s: %s", subset, e)
            self.log_text.append(f"Błąd podczas pobierania {subset}: {e}")

    def download_all(self):
        if not self.dataset_name:
            self.log_text.append("Proszę wybrać dataset!")
            return

        try:
            response = requests.get(
                f"{self.api_url}/download_dataset/{self.username}/{self.dataset_name}",
                stream=True
            )
            response.raise_for_status()
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Zapisz wyniki", f"{self.dataset_name}_results.zip", "ZIP files (*.zip)"
            )
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                self.log_text.append(f"Pobrano cały dataset do: {save_path}")
                requests.delete(f"{self.api_url}/delete_zip/{self.username}/{self.dataset_name}")
            else:
                self.log_text.append("Anulowano zapisywanie wyników.")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas pobierania całego datasetu: %s", e)
            self.log_text.append(f"Błąd podczas pobierania datasetu: {e}")