"""
Implementacja zakładki zarządzania użytkownikami w aplikacji SMLE.

Moduł dostarcza interfejs użytkownika do zarządzania kontami użytkowników 
w systemie, w tym przeglądania listy użytkowników, tworzenia, edycji, 
blokowania i odblokowywania kont. Edycja użytkownika odbywa się w osobnym 
oknie dialogowym. Zakładka jest dostępna tylko dla użytkowników z rolą 
administratora.
"""
#######################
# Importy bibliotek
#######################
from PyQt5 import QtWidgets, QtGui, QtCore
import requests
import logging

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EditUserDialog(QtWidgets.QDialog):
    """
    Okno dialogowe do edycji danych użytkownika.
    """
    def __init__(self, user, api_url, parent=None):
        super().__init__(parent)
        self.user = user
        self.api_url = api_url
        self.setWindowTitle("Edytuj użytkownika")
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        form_layout = QtWidgets.QFormLayout()

        # Pole na nazwę użytkownika
        self.name_input = QtWidgets.QLineEdit(self.user["name"])
        self.name_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #948979;
                padding: 5px;
            }
        """)
        form_layout.addRow("Nazwa użytkownika:", self.name_input)

        # Pole na hasło
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setPlaceholderText("Nowe hasło (opcjonalne)")
        self.password_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #948979;
                padding: 5px;
            }
        """)
        form_layout.addRow("Hasło:", self.password_input)

        # Przyciski
        button_layout = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QPushButton("Zapisz")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #222831;
                color: #DFD0B8;
                border: 1px solid #948979;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #393E46;
            }
        """)
        save_button.clicked.connect(self.save_changes)

        cancel_button = QtWidgets.QPushButton("Anuluj")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #222831;
                color: #DFD0B8;
                border: 1px solid #948979;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #393E46;
            }
        """)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def save_changes(self):
        username = self.name_input.text().strip()
        password = self.password_input.text().strip()

        if not username:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nazwa użytkownika jest wymagana!")
            return

        data = {"username": username}
        if password:
            data["password"] = password

        logger.debug(f"[DEBUG] Edycja użytkownika ID {self.user['id']}: {data}")
        try:
            response = requests.put(f"{self.api_url}/users/{self.user['id']}", json=data)
            response.raise_for_status()
            QtWidgets.QMessageBox.information(self, "Sukces", "Dane użytkownika zaktualizowane!")
            self.accept()
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Błąd podczas edycji użytkownika: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się zaktualizować danych: {e}")

class UsersTab(QtWidgets.QWidget):
    """
    Zakładka zarządzania użytkownikami w aplikacji SMLE.
    
    Wyświetla tabelę z listą użytkowników systemu, ich danymi oraz przyciski
    do zarządzania (blokowanie, edycja). Umożliwia tworzenie nowych kont
    i edytowanie istniejących w osobnym oknie dialogowym. Status 
    użytkownika jest kolorowany: zielony dla 'active', czerwony dla 'inactive'.
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
        - Tabela wyświetlająca użytkowników z kolumną 'Zmiany'
        - Formularz do tworzenia nowych użytkowników
        """
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tworzymy tabelę do wyświetlania użytkowników
        self.users_table = QtWidgets.QTableWidget()
        self.users_table.setRowCount(0)
        self.users_table.setColumnCount(6)  # Nazwa, Data rejestracji, Ostatnie logowanie, Status, Rola, Zmiany

        # Ustawiamy nagłówki kolumn
        self.users_table.setHorizontalHeaderLabels([
            "Nazwa Użytkownika", "Data Rejestracji", 
            "Ostatnie Logowanie", "Status", "Rola", "Zmiany"
        ])

        # Zablokowanie edytowania danych w tabeli
        self.users_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Ustawienie szerokości kolumn
        self.users_table.setColumnWidth(0, 150)  # Nazwa użytkownika
        self.users_table.setColumnWidth(1, 150)  # Data rejestracji
        self.users_table.setColumnWidth(2, 150)  # Ostatnie logowanie
        self.users_table.setColumnWidth(3, 100)  # Status
        self.users_table.setColumnWidth(4, 100)  # Rola
        self.users_table.setColumnWidth(5, 250)  # Zmiany

        # Dodanie tabeli do głównego układu
        layout.addWidget(self.users_table)

        # Formularz do dodawania użytkownika
        form_layout = QtWidgets.QFormLayout()
        
        # Pola formularza
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #948979;
                padding: 5px;
            }
        """)
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #948979;
                padding: 5px;
            }
        """)
        
        # Przycisk dodawania
        self.add_user_button = QtWidgets.QPushButton("Dodaj użytkownika")
        self.add_user_button.setStyleSheet("""
            QPushButton {
                background-color: #222831;
                color: #FFFFFF;
                border: 1px solid #948979;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #393E46;
            }
        """)
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
        Status 'active' jest zielony, 'inactive' czerwony. Dodaje przyciski w kolumnie 'Zmiany'.
        
        Args:
            users (list): Lista słowników zawierających dane użytkowników
        """
        self.users_table.setRowCount(len(users))
        for row, user in enumerate(users):
            self.users_table.setItem(row, 0, QtWidgets.QTableWidgetItem(user["name"]))
            self.users_table.setItem(row, 1, QtWidgets.QTableWidgetItem(user["register_date"]))
            self.users_table.setItem(row, 2, QtWidgets.QTableWidgetItem(user.get("last_login", "Brak danych")))
            
            # Kolorowanie statusu
            status_item = QtWidgets.QTableWidgetItem(user["status"])
            if user["status"] == "active":
                status_item.setForeground(QtGui.QColor("green"))
            else:
                status_item.setForeground(QtGui.QColor("red"))
            self.users_table.setItem(row, 3, status_item)
            
            self.users_table.setItem(row, 4, QtWidgets.QTableWidgetItem(user["role"]))
            
            # Widget dla kolumny "Zmiany"
            button_widget = QtWidgets.QWidget()
            button_layout = QtWidgets.QHBoxLayout()
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.setSpacing(5)
            
            # Przycisk Zablokuj/Odblokuj
            block_button = QtWidgets.QPushButton("Zablokuj" if user["status"] == "active" else "Odblokuj")
            block_button.setStyleSheet("""
                QPushButton {
                    background-color: #222831;
                    color: #FFFFFF;
                    border: 1px solid #948979;
                    padding: 3px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #393E46;
                }
            """)
            block_button.clicked.connect(lambda _, u=user: self.toggle_user_status(u["id"], u["status"]))
            
            # Przycisk Edytuj
            edit_button = QtWidgets.QPushButton("Edytuj")
            edit_button.setStyleSheet("""
                QPushButton {
                    background-color: #222831;
                    color: #FFFFFF;
                    border: 1px solid #948979;
                    padding: 3px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #393E46;
                }
            """)
            edit_button.clicked.connect(lambda _, u=user: self.open_edit_dialog(u))
            
            button_layout.addWidget(block_button)
            button_layout.addWidget(edit_button)
            button_widget.setLayout(button_layout)
            self.users_table.setCellWidget(row, 5, button_widget)

    def open_edit_dialog(self, user):
        """
        Otwiera okno dialogowe do edycji użytkownika.
        
        Args:
            user (dict): Dane użytkownika
        """
        dialog = EditUserDialog(user, self.api_url, self)
        if dialog.exec_():
            self.load_users()

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

    def toggle_user_status(self, user_id, current_status):
        """
        Przełącza status użytkownika (active/inactive).
        
        Args:
            user_id (int): ID użytkownika
            current_status (str): Aktualny status użytkownika
        """
        new_status = "inactive" if current_status == "active" else "active"
        logger.debug(f"[DEBUG] Przełączanie statusu użytkownika ID {user_id} na {new_status}")
        try:
            response = requests.put(f"{self.api_url}/users/{user_id}/status", json={"status": new_status})
            response.raise_for_status()
            QtWidgets.QMessageBox.information(self, "Sukces", f"Status użytkownika zmieniony na {new_status}!")
            self.load_users()
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Błąd podczas zmiany statusu: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się zmienić statusu: {e}")