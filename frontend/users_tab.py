import sys
import requests
from PyQt5 import QtWidgets, QtCore

class UsersTab(QtWidgets.QWidget):
    def __init__(self, api_url):
        super().__init__()
        self.api_url = api_url
        self.init_ui()
        self.load_users()  # Załaduj użytkowników przy starcie

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tworzymy tabelę do wyświetlania użytkowników
        self.users_table = QtWidgets.QTableWidget()
        self.users_table.setRowCount(0)  # Zaczynamy od pustej tabeli
        self.users_table.setColumnCount(5)  # Dodajemy kolumnę "Rola"

        # Ustawiamy nagłówki kolumn
        self.users_table.setHorizontalHeaderLabels(["Nazwa Użytkownika", "Data Rejestracji", "Ostatnie Logowanie", "Status", "Rola"])

        # Zablokowanie edytowania danych
        self.users_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Ustawienie szerokości kolumn
        self.users_table.setColumnWidth(0, 150)
        self.users_table.setColumnWidth(1, 150)
        self.users_table.setColumnWidth(2, 150)
        self.users_table.setColumnWidth(3, 100)
        self.users_table.setColumnWidth(4, 100)  # Kolumna "Rola"

        layout.addWidget(self.users_table)

        # Formularz do dodawania użytkownika
        form_layout = QtWidgets.QFormLayout()
        self.name_input = QtWidgets.QLineEdit()
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.add_user_button = QtWidgets.QPushButton("Dodaj użytkownika")
        self.add_user_button.clicked.connect(self.create_user)

        form_layout.addRow("Nazwa użytkownika:", self.name_input)
        form_layout.addRow("Hasło:", self.password_input)
        form_layout.addWidget(self.add_user_button)

        layout.addLayout(form_layout)
        self.setLayout(layout)

    def load_users(self):
        """Pobiera użytkowników z API backendu i wyświetla w tabeli"""
        try:
            response = requests.get(f"{self.api_url}/users")  # Połączenie z backendem
            response.raise_for_status()  # Sprawdzenie, czy odpowiedź jest poprawna

            users = response.json()
            self.display_users(users)  # 🔥 Wywołujemy funkcję wyświetlania danych
        except requests.exceptions.RequestException as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")

    def display_users(self, users):
        """Wypełnia tabelę użytkownikami"""
        self.users_table.setRowCount(len(users))
        for row, user in enumerate(users):
            self.users_table.setItem(row, 0, QtWidgets.QTableWidgetItem(user["name"]))
            self.users_table.setItem(row, 1, QtWidgets.QTableWidgetItem(user["register_date"]))
            self.users_table.setItem(row, 2, QtWidgets.QTableWidgetItem(user.get("last_login", "Brak danych")))
            self.users_table.setItem(row, 3, QtWidgets.QTableWidgetItem(user["status"]))
            self.users_table.setItem(row, 4, QtWidgets.QTableWidgetItem(user["role"]))  # Dodanie kolumny "Rola"
    
    def create_user(self):
        """Wysyła dane nowego użytkownika do backendu"""
        username = self.name_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wypełnij wszystkie pola!")
            return

        try:
            response = requests.post(f"{self.api_url}/users", json={"username": username, "password": password})
            response.raise_for_status()
            QtWidgets.QMessageBox.information(self, "Sukces", "Użytkownik dodany!")
            self.load_users()  # Odśwież listę użytkowników
        except requests.exceptions.RequestException as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się dodać użytkownika: {e}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UsersTab()
    window.show()
    sys.exit(app.exec_())
