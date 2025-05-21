"""
Implementacja zakładki Archiwum w aplikacji SMLE.

Moduł dostarcza interfejs użytkownika do przeglądania historii operacji 
wykonywanych na modelach, umożliwiając śledzenie zmian i aktywności w systemie.
"""
#######################
# Importy bibliotek
#######################
from PyQt5 import QtWidgets
import requests

class ArchiveTab(QtWidgets.QWidget):
    """
    Zakładka Archiwum wyświetlająca historię operacji na modelach.
    
    Komponent zawiera tabelę z zapisem operacji wykonywanych przez użytkowników,
    wraz z informacjami o dacie, modelu i typie operacji. Umożliwia także
    podgląd szczegółów wybranych wpisów.
    """
    def __init__(self, api_url):
        """
        Inicjalizuje zakładkę Archiwum i pobiera dane początkowe.
        
        Args:
            api_url (str): Bazowy adres URL API backendu
        """
        super().__init__()
        self.init_ui()
        self.api_url = api_url
        self.load_archive_data()  # Załaduj dane z archiwum przy starcie

    def init_ui(self):
        """
        Tworzy i konfiguruje elementy interfejsu użytkownika zakładki.
        
        Komponenty:
        - Tabela wyświetlająca dane archiwalne
        - Przycisk do podglądu szczegółów wybranego wpisu
        """
        # Główny layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tabela dla Archiwum
        self.archive_table = QtWidgets.QTableWidget()
        self.archive_table.setRowCount(0)  # Rozpoczynamy od pustej tabeli
        self.archive_table.setColumnCount(4)  # 4 kolumny: Model, Operacja, Data, Użytkownik

        # Ustawiamy nagłówki kolumn
        self.archive_table.setHorizontalHeaderLabels(["Model", "Operacja", "Data", "Użytkownik"])

        # Zablokowanie edytowania danych
        self.archive_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

        # Pobieramy nagłówek tabeli
        header = self.archive_table.horizontalHeader()

        # Zablokowanie zmiany rozmiaru kolumn
        header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        # Ustawienie stałej szerokości dla każdej kolumny
        self.archive_table.setColumnWidth(0, 400)
        self.archive_table.setColumnWidth(1, 200)
        self.archive_table.setColumnWidth(2, 200)
        self.archive_table.setColumnWidth(3, 150)

        # Włączamy rozciąganie tabeli, aby zajmowała całą przestrzeń
        self.archive_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Dodajemy tabelę do layoutu
        layout.addWidget(self.archive_table, stretch=1)

        # Układ przycisków
        # btn_layout = QtWidgets.QHBoxLayout()
        # self.view_details_btn = QtWidgets.QPushButton("Zobacz szczegóły")
        # btn_layout.addWidget(self.view_details_btn)

        # # Dodajemy przyciski do layoutu
        # layout.addLayout(btn_layout)

        self.setLayout(layout)

    def load_archive_data(self):
        """
        Pobiera dane z archiwum z API backendu i inicjuje ich wyświetlenie.
        
        Łączy się z endpointem /archives, pobiera dane w formacie JSON
        i przekazuje je do funkcji wyświetlającej.
        
        W przypadku błędu komunikacji wyświetla odpowiedni komunikat.
        """
        try:
            response = requests.get(f"{self.api_url}/archives")
            response.raise_for_status()  # Sprawdzenie, czy odpowiedź jest poprawna

            archive_data = response.json()
            self.display_archive_data(archive_data)
        except requests.exceptions.RequestException as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")

    def display_archive_data(self, archive_data):
        """
        Wypełnia tabelę danymi z archiwum.
        
        Args:
            archive_data (list): Lista słowników z danymi archiwalnymi
                                 zawierającymi pola: model_display_name, action, date, username
        """
        self.archive_table.setRowCount(len(archive_data))
        for row, record in enumerate(archive_data):
            # Przypisanie danych do odpowiednich kolumn
            self.archive_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(record["model_display_name"])))  # Model
            self.archive_table.setItem(row, 1, QtWidgets.QTableWidgetItem(record["action"]))  # Operacja
            self.archive_table.setItem(row, 2, QtWidgets.QTableWidgetItem(record["date"]))  # Data
            self.archive_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(record["username"])))  # Użytkownik
            
    def view_details(self):
        """
        Wyświetla szczegóły wybranego wpisu archiwalnego.
        
        Funkcja placeholderowa - w pełnej implementacji mogłaby otwierać
        okno dialogowe z dodatkowymi informacjami o wybranej operacji.
        """
        QtWidgets.QMessageBox.information(self, "Szczegóły", "Wyświetlanie szczegółów operacji...")