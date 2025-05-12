"""
Implementacja zakładki zarządzania modelami w aplikacji SMLE.

Moduł dostarcza interfejs użytkownika umożliwiający przeglądanie, wczytywanie
i usuwanie modeli uczenia maszynowego wykorzystywanych w aplikacji. Wyświetla
listę dostępnych modeli wraz z ich szczegółowymi parametrami oraz umożliwia
podstawowe operacje zarządzania.
"""
#######################
# Importy lokalne
#######################
from PyQt5 import QtWidgets
import requests
import os
import logging

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelsTab(QtWidgets.QWidget):
    """
    Zakładka zarządzania modelami uczenia maszynowego.
    
    Wyświetla tabelę zawierającą modele dostępne w systemie wraz z ich parametrami
    oraz udostępnia funkcje zarządzania modelami, takie jak: wczytywanie nowych modeli,
    usuwanie istniejących oraz przekierowanie do tworzenia nowych modeli.
    """
    def __init__(self, user_name, api_url):
        """
        Inicjalizuje zakładkę modeli.
        
        Args:
            user_name (str): Nazwa zalogowanego użytkownika
            api_url (str): Adres URL API backendu
        """
        super().__init__()
        self.user_name = user_name
        self.api_url = api_url
        logger.debug(f"[DEBUG] Inicjalizacja ModelsTab: user={user_name}, api={api_url}")
        self.init_ui()
        self.load_models()

    def init_ui(self):
        """
        Tworzy i konfiguruje elementy interfejsu użytkownika zakładki.
        
        Komponenty:
        - Tabela wyświetlająca szczegółowe informacje o modelach
        - Przyciski do operacji zarządzania modelami (wczytaj, usuń, utwórz)
        """
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tabela modeli
        self.models_table = QtWidgets.QTableWidget()
        self.models_table.setRowCount(0)
        self.models_table.setColumnCount(8)
        self.models_table.setHorizontalHeaderLabels([
            "ID", "Nazwa Modelu", "Algorytm", "Wersja", "Dokładność",
            "Data Utworzenia", "Status", "Data Treningu"
        ])
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.models_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.models_table.setColumnWidth(0, 50)
        self.models_table.setColumnWidth(1, 250)
        self.models_table.setColumnWidth(2, 150)
        self.models_table.setColumnWidth(3, 100)
        self.models_table.setColumnWidth(4, 100)
        self.models_table.setColumnWidth(5, 150)
        self.models_table.setColumnWidth(6, 100)
        self.models_table.setColumnWidth(7, 150)

        layout.addWidget(self.models_table)

        # Panel przycisków
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.load_model_btn = QtWidgets.QPushButton("Wczytaj model")
        self.load_model_btn.clicked.connect(self.show_load_model_dialog)
        
        self.delete_model_btn = QtWidgets.QPushButton("Usuń model")
        self.delete_model_btn.clicked.connect(self.delete_selected_model)

        self.create_new_model_btn = QtWidgets.QPushButton("Utwórz Nowy Model")
        self.create_new_model_btn.clicked.connect(self.create_new_model)

        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.delete_model_btn)
        btn_layout.addWidget(self.create_new_model_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_models(self):
        """
        Pobiera listę dostępnych modeli z API backendu i wyświetla je w tabeli.
        
        W przypadku błędu komunikacji z serwerem wyświetla komunikat ostrzegawczy.
        """
        logger.debug(f"[DEBUG] Pobieranie modeli z {self.api_url}/models")
        try:
            response = requests.get(f"{self.api_url}/models")
            response.raise_for_status()
            models = response.json()
            self.display_models(models)
            logger.debug(f"[DEBUG] Pobrano {len(models)} modeli")
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Nie udało się pobrać modeli: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")

    def display_models(self, models):
        """
        Wypełnia tabelę danymi modeli.
        
        Dla każdego modelu wyświetla jego parametry w odpowiedniej komórce tabeli.
        
        Args:
            models (list): Lista słowników zawierających dane modeli
        """
        self.models_table.setRowCount(len(models))
        for row, model in enumerate(models):
            training_date = model.get("training_date", "")
            self.models_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(model["id"])))
            self.models_table.setItem(row, 1, QtWidgets.QTableWidgetItem(model["name"]))
            self.models_table.setItem(row, 2, QtWidgets.QTableWidgetItem(model["algorithm"]))
            self.models_table.setItem(row, 3, QtWidgets.QTableWidgetItem(model["version"]))
            self.models_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(model["accuracy"])))
            self.models_table.setItem(row, 5, QtWidgets.QTableWidgetItem(model["creation_date"]))
            self.models_table.setItem(row, 6, QtWidgets.QTableWidgetItem(model["status"]))
            self.models_table.setItem(row, 7, QtWidgets.QTableWidgetItem(training_date))

    def create_new_model(self):
        """
        Obsługuje tworzenie nowego modelu.
        
        Obecnie funkcja tylko informuje o przekierowaniu do zakładki treningowej,
        gdzie odbywa się właściwy proces tworzenia modelu.
        """
        logger.debug("[DEBUG] Wywoływanie create_new_model()")
        QtWidgets.QMessageBox.information(self, "Informacja", 
                                         "Aby utworzyć nowy model, przejdź do zakładki 'Trening'.")

    def delete_selected_model(self):
        """
        Usuwa wybrany model po potwierdzeniu przez użytkownika.
        
        Weryfikuje, czy model został wybrany, a następnie wysyła żądanie
        usunięcia do API backendu. Aktualizuje tabelę po pomyślnym usunięciu.
        """
        selected_row = self.models_table.currentRow()
        if selected_row < 0:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano żadnego modelu do usunięcia.")
            return

        model_id_item = self.models_table.item(selected_row, 0)
        if not model_id_item:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie można odczytać ID modelu.")
            return

        model_id = model_id_item.text()

        confirm = QtWidgets.QMessageBox.question(
            self,
            "Potwierdzenie",
            f"Czy na pewno chcesz usunąć model o ID {model_id}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if confirm == QtWidgets.QMessageBox.Yes:
            logger.debug(f"[DEBUG] Usuwanie modelu o ID {model_id}")
            try:
                response = requests.delete(f"{self.api_url}/models/{model_id}")
                response.raise_for_status()
                QtWidgets.QMessageBox.information(self, "Sukces", "Model został usunięty.")
                self.load_models()
            except requests.exceptions.RequestException as e:
                logger.error(f"[ERROR] Błąd podczas usuwania modelu: {e}")
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Nie udało się usunąć modelu: {e}")

    def show_load_model_dialog(self):
        """
        Wyświetla dialog umożliwiający wczytanie zewnętrznego modelu.
        
        Dialog zawiera:
        - Wybór algorytmu dla wczytywanego modelu
        - Pole wyboru pliku modelu (.pth)
        - Przyciski potwierdzenia i anulowania operacji
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Wczytaj model")
        dialog_layout = QtWidgets.QVBoxLayout(dialog)

        # Algorytm
        algo_label = QtWidgets.QLabel("Wybierz algorytm:")
        algo_combo = QtWidgets.QComboBox()
        algo_combo.addItems(["Mask R-CNN", "FasterRCNN", "MCNN"])
        dialog_layout.addWidget(algo_label)
        dialog_layout.addWidget(algo_combo)

        # Wybór pliku
        file_layout = QtWidgets.QHBoxLayout()
        file_path_input = QtWidgets.QLineEdit()
        file_path_input.setPlaceholderText("Ścieżka do pliku .pth")
        browse_btn = QtWidgets.QPushButton("Przeglądaj")

        def browse_file():
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz plik modelu", "", "Model files (*.pth)")
            if file_path:
                file_path_input.setText(file_path)

        browse_btn.clicked.connect(browse_file)
        file_layout.addWidget(file_path_input)
        file_layout.addWidget(browse_btn)
        dialog_layout.addLayout(file_layout)

        # OK / Anuluj
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        dialog_layout.addWidget(button_box)

        def on_accept():
            algorithm = algo_combo.currentText()
            file_path = file_path_input.text()
            if not file_path or not os.path.isfile(file_path):
                QtWidgets.QMessageBox.warning(dialog, "Błąd", "Nie wybrano prawidłowego pliku modelu.")
                return
            dialog.accept()
            self.load_model(algorithm, file_path)

        button_box.accepted.connect(on_accept)
        button_box.rejected.connect(dialog.reject)
        dialog.exec_()

    def load_model(self, algorithm, file_path):
        """
        Wczytuje wybrany plik modelu i przesyła go do API backendu.
        
        Args:
            algorithm (str): Nazwa algorytmu, dla którego wczytywany jest model
            file_path (str): Ścieżka do pliku modelu (.pth)
        """
        logger.debug(f"[DEBUG] Wczytywanie modelu: algorytm={algorithm}, plik={file_path}")
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {
                    'algorithm': algorithm,
                    'user_name': self.user_name
                }
                # Naprawiony URL - dodane f przed stringiem dla formatowania
                response = requests.post(f"{self.api_url}/models/upload", files=files, data=data)
                response.raise_for_status()
                QtWidgets.QMessageBox.information(self, "Sukces", "Model został wczytany.")
                self.load_models()
        except Exception as e:
            logger.error(f"[ERROR] Błąd podczas wczytywania modelu: {e}")
            QtWidgets.QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać modelu: {e}")