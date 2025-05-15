"""
Implementacja zakładki zarządzania użytkownikami w aplikacji SMLE.

Moduł dostarcza interfejs użytkownika do zarządzania kontami użytkowników 
w systemie, w tym przeglądania listy użytkowników oraz tworzenia nowych kont.
Zakładka jest dostępna tylko dla użytkowników z rolą administratora.
"""
#######################
# Importy bibliotek
#######################
from PyQt5 import QtWidgets
import requests
import logging

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UsersTab(QtWidgets.QWidget):
    """
    Zakładka zarządzania użytkownikami w aplikacji SMLE.
    
    Wyświetla tabelę z listą użytkowników systemu oraz udostępnia formularz
    do tworzenia nowych kont. Pozwala administratorom na monitorowanie
    użytkowników i zarządzanie dostępem do systemu.
    """
    def __init__(self, api_url):
        """
        Inicjalizuje zakładkę zarządzania użytkownikami.
        
        Args:
            api_url (str): Adres URL API backendu
        """
        super().__init__()
        self.api_url = api_url
        logger.debug(f"[DEBUG] Inicjalizacja UsersTab: api_url={self.api_url}")
        self.init_ui()
        self.load_users()  # Załaduj użytkowników przy starcie

    def init_ui(self):
        """
        Tworzy i konfiguruje elementy interfejsu użytkownika zakładki.
        
        Komponenty:
        - Tabela wyświetlająca użytkowników i ich dane
        - Formularz do tworzenia nowych użytkowników
        """
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tworzymy tabelę do wyświetlania użytkowników
        self.users_table = QtWidgets.QTableWidget()
        self.users_table.setRowCount(0)  # Zaczynamy od pustej tabeli
        self.users_table.setColumnCount(5)  # ID, Nazwa, Data rejestracji, Ostatnie logowanie, Status, Rola

        # Ustawiamy nagłówki kolumn
        self.users_table.setHorizontalHeaderLabels([
            "Nazwa Użytkownika", "Data Rejestracji", 
            "Ostatnie Logowanie", "Status", "Rola"
        ])

        # Zablokowanie edytowania danych w tabeli
        self.users_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Ustawienie szerokości kolumn
        self.users_table.setColumnWidth(0, 150)  # Nazwa użytkownika
        self.users_table.setColumnWidth(1, 150)  # Data rejestracji
        self.users_table.setColumnWidth(2, 150)  # Ostatnie logowanie
        self.users_table.setColumnWidth(3, 100)  # Status
        self.users_table.setColumnWidth(4, 100)  # Rola

        # Dodanie tabeli do głównego układu
        layout.addWidget(self.users_table)

        # Formularz do dodawania użytkownika
        form_layout = QtWidgets.QFormLayout()
        
        # Pola formularza
        self.name_input = QtWidgets.QLineEdit()
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        
        # Przycisk dodawania
        self.add_user_button = QtWidgets.QPushButton("Dodaj użytkownika")
        self.add_user_button.clicked.connect(self.create_user)

        # Układ formularza
        form_layout.addRow("Nazwa użytkownika:", self.name_input)
        form_layout.addRow("Hasło:", self.password_input)
        form_layout.addWidget(self.add_user_button)

        # Dodanie formularza do głównego układu
        layout.addLayout(form_layout)
        self.setLayout(layout)

    def load_users(self):
        """
        Pobiera listę użytkowników z API backendu i wyświetla w tabeli.
        
        W przypadku błędu komunikacji z serwerem wyświetla komunikat ostrzegawczy.
        """
        logger.debug(f"[DEBUG] Pobieranie użytkowników z {self.api_url}/users")
        try:
            response = requests.get(f"{self.api_url}/users")
            response.raise_for_status()
            users = response.json()
            self.display_users(users)
            logger.debug(f"[DEBUG] Pobrano {len(users)} użytkowników")
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Błąd podczas pobierania użytkowników: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")

    def display_users(self, users):
        """
        Wypełnia tabelę danymi użytkowników.
        
        Dla każdego użytkownika wyświetla jego parametry w odpowiedniej komórce tabeli.
        
        Args:
            users (list): Lista słowników zawierających dane użytkowników
        """
        self.users_table.setRowCount(len(users))
        for row, user in enumerate(users):
            self.users_table.setItem(row, 0, QtWidgets.QTableWidgetItem(user["name"]))
            self.users_table.setItem(row, 1, QtWidgets.QTableWidgetItem(user["register_date"]))
            self.users_table.setItem(row, 2, QtWidgets.QTableWidgetItem(user.get("last_login", "Brak danych")))
            self.users_table.setItem(row, 3, QtWidgets.QTableWidgetItem(user["status"]))
            self.users_table.setItem(row, 4, QtWidgets.QTableWidgetItem(user["role"]))
    
    def create_user(self):
        """
        Tworzy nowego użytkownika w systemie.
        
        Pobiera dane z formularza, waliduje je i wysyła żądanie utworzenia
        użytkownika do API backendu. Po pomyślnym utworzeniu odświeża listę użytkowników.
        """
        username = self.name_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wypełnij wszystkie pola!")
            return

        logger.debug(f"[DEBUG] Tworzenie użytkownika: {username}")
        try:
            data = {
                "username": username, 
                "password": password
            }
            response = requests.post(f"{self.api_url}/users", json=data)
            response.raise_for_status()
            
            QtWidgets.QMessageBox.information(self, "Sukces", "Użytkownik dodany!")
            
            # Wyczyść pola formularza
            self.name_input.clear()
            self.password_input.clear()
            
            # Odśwież listę użytkowników
            self.load_users()
            logger.debug(f"[DEBUG] Użytkownik {username} został utworzony")
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Błąd podczas tworzenia użytkownika: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się dodać użytkownika: {e}")