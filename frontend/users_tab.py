import sys
import requests
from PyQt5 import QtWidgets, QtCore

class UsersTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_users()  # Załaduj użytkowników przy starcie

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tworzymy tabelę do wyświetlania użytkowników
        self.users_table = QtWidgets.QTableWidget()
        self.users_table.setRowCount(0)  # Zaczynamy od pustej tabeli
        self.users_table.setColumnCount(4)  # 4 kolumny: Nazwa Użytkownika, Data Rejestracji, Ostatnie Logowanie, Status

        # Ustawiamy nagłówki kolumn
        self.users_table.setHorizontalHeaderLabels(["Nazwa Użytkownika", "Data Rejestracji", "Ostatnie Logowanie", "Status"])

        # Zablokowanie edytowania danych
        self.users_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Zablokowanie zmiany rozmiaru kolumn
        self.users_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.users_table.setColumnWidth(0, 200)
        self.users_table.setColumnWidth(1, 200)
        self.users_table.setColumnWidth(2, 200)
        self.users_table.setColumnWidth(3, 150)

        # Dodajemy tabelę do layoutu
        layout.addWidget(self.users_table)

        self.setLayout(layout)

    def load_users(self):
        """Pobiera użytkowników z API backendu i wyświetla w tabeli"""
        try:
            response = requests.get("http://localhost:8000/users")  # Połączenie z backendem
            response.raise_for_status()  # Sprawdzenie, czy odpowiedź jest poprawna

            users = response.json()

            # Dodanie logowania, aby sprawdzić strukturę danych
            print("Odpowiedź z backendu:", users)

            self.display_users(users)

        except requests.exceptions.RequestException as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")


    def display_users(self, users):
        """Wypełnia tabelę użytkownikami"""
        self.users_table.setRowCount(len(users))  # Ustawiamy liczbę wierszy na liczbę użytkowników
        for row, user in enumerate(users):
            # Zmieniamy "username" na "name"
            self.users_table.setItem(row, 0, QtWidgets.QTableWidgetItem(user["name"]))  # Kolumna 0 to nazwa użytkownika

            # Zmieniamy "registration_date" na "register_date" (lub jak jest w odpowiedzi z API)
            self.users_table.setItem(row, 1, QtWidgets.QTableWidgetItem(user["register_date"]))

            # Zmieniamy "last_login" na odpowiednią nazwę (jeśli taka kolumna jest w odpowiedzi z backendu)
            self.users_table.setItem(row, 2, QtWidgets.QTableWidgetItem(user.get("last_login", "Brak danych")))  # Używamy get, aby uniknąć błędu, gdy brak danych

            # Zmieniamy "status" na odpowiednią nazwę (w backendzie jest to "status")
            self.users_table.setItem(row, 3, QtWidgets.QTableWidgetItem(user["status"]))
