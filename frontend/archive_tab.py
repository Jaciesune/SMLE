from PyQt5 import QtWidgets
import requests

class ArchiveTab(QtWidgets.QWidget):
    def __init__(self, api_url):
        super().__init__()
        self.init_ui()
        self.api_url = api_url
        self.load_archive_data()  # Załaduj dane z archiwum przy starcie

    def init_ui(self):
        # Główny layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tabela dla Archiwum
        self.archive_table = QtWidgets.QTableWidget()
        self.archive_table.setRowCount(0)  # Rozpoczynamy od pustej tabeli
        self.archive_table.setColumnCount(4)  # 5 kolumn: Model, Algorytm, Operacja, Data, Użytkownik

        # Ustawiamy nagłówki kolumn
        self.archive_table.setHorizontalHeaderLabels(["Model", "Operacja", "Data", "Użytkownik"])

        # Zablokowanie edytowania danych
        self.archive_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

        # Pobieramy nagłówek tabeli
        header = self.archive_table.horizontalHeader()

        # Zablokowanie zmiany rozmiaru kolumn
        header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        # Ustawienie stałej szerokości dla każdej kolumny
        self.archive_table.setColumnWidth(0, 200)
        self.archive_table.setColumnWidth(1, 200)
        self.archive_table.setColumnWidth(2, 200)
        self.archive_table.setColumnWidth(3, 200)

        # Włączamy rozciąganie tabeli, aby zajmowała całą przestrzeń
        self.archive_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Dodajemy tabelę do layoutu
        layout.addWidget(self.archive_table, stretch=1)

        # Układ przycisków
        btn_layout = QtWidgets.QHBoxLayout()
        self.view_details_btn = QtWidgets.QPushButton("Zobacz szczegóły")
        btn_layout.addWidget(self.view_details_btn)

        # Dodajemy przycisk do layoutu
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def load_archive_data(self):
        """Pobiera dane z archiwum z API backendu i wyświetla je w tabeli"""
        try:
            response = requests.get(f"{self.api_url}/archives")  # Poprawiony URL API
            response.raise_for_status()  # Sprawdzenie, czy odpowiedź jest poprawna

            archive_data = response.json()
            self.display_archive_data(archive_data)  # 🔥 Wywołujemy funkcję wyświetlania danych
        except requests.exceptions.RequestException as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")

    def display_archive_data(self, archive_data):
        """Wypełnia tabelę danymi z archiwum"""
        self.archive_table.setRowCount(len(archive_data))
        for row, record in enumerate(archive_data):
            # Przypisanie danych do odpowiednich kolumn
            # Należy założyć, że backend zwrócił również `model_name` i `user_name` na podstawie `model_id` i `user_id`
            self.archive_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(record["model_id"])))  # Model
            self.archive_table.setItem(row, 1, QtWidgets.QTableWidgetItem(record["action"]))  # Operacja
            self.archive_table.setItem(row, 2, QtWidgets.QTableWidgetItem(record["date"]))  # Data
            self.archive_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(record["user_id"])))  # Użytkownik
            
    def view_details(self):
        """Funkcja przenosząca do szczegółów (implementacja zależna od wymagań)"""
        QtWidgets.QMessageBox.information(self, "Szczegóły", "Wyświetlanie szczegółów operacji...")

