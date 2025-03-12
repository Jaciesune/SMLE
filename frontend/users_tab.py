from PyQt5 import QtWidgets, QtCore

class UsersTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tworzymy tabelę do wyświetlania użytkowników
        self.users_table = QtWidgets.QTableWidget()
        self.users_table.setRowCount(2)  # Dodajemy 2 wiersze (przykładowi użytkownicy)
        self.users_table.setColumnCount(4)  # 4 kolumny: Nazwa Użytkownika, Data Rejestracji, Ostatnie Logowanie, Status

        # Ustawiamy nagłówki kolumn
        self.users_table.setHorizontalHeaderLabels(["Nazwa Użytkownika", "Data Rejestracji", "Ostatnie Logowanie", "Status"])

        # Wypełniamy tabelę przykładowymi danymi
        self.users_table.setItem(0, 0, QtWidgets.QTableWidgetItem("Uzytkownik_TEST"))
        self.users_table.setItem(0, 1, QtWidgets.QTableWidgetItem("09:33 07.03.2025"))
        self.users_table.setItem(0, 2, QtWidgets.QTableWidgetItem("08:11 08.03.2025"))
        self.users_table.setItem(0, 3, QtWidgets.QTableWidgetItem("Aktywne"))

        self.users_table.setItem(1, 0, QtWidgets.QTableWidgetItem("Uzytkownik_TEST2"))
        self.users_table.setItem(1, 1, QtWidgets.QTableWidgetItem("09:45 08.03.2025"))
        self.users_table.setItem(1, 2, QtWidgets.QTableWidgetItem("09:50 08.03.2025"))
        self.users_table.setItem(1, 3, QtWidgets.QTableWidgetItem("Aktywne"))

        # Zablokowanie edytowania danych
        self.users_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Zablokowanie zmiany rozmiaru kolumn
        self.users_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.users_table.setColumnWidth(0, 200)  # Szerokość pierwszej kolumny
        self.users_table.setColumnWidth(1, 200)  # Szerokość drugiej kolumny
        self.users_table.setColumnWidth(2, 200)  # Szerokość trzeciej kolumny
        self.users_table.setColumnWidth(3, 150)  # Szerokość czwartej kolumny

        # Dodajemy tabelę do layoutu
        layout.addWidget(self.users_table)

        # Formularz do rejestracji nowego użytkownika
        form_layout = QtWidgets.QFormLayout()
        self.new_username = QtWidgets.QLineEdit()
        self.new_password = QtWidgets.QLineEdit()
        self.new_password.setEchoMode(QtWidgets.QLineEdit.Password)
        form_layout.addRow("Nazwa użytkownika:", self.new_username)
        form_layout.addRow("Hasło:", self.new_password)
        layout.addLayout(form_layout)

        # Przycisk rejestracji
        self.register_btn = QtWidgets.QPushButton("Zarejestruj użytkownika")
        self.register_btn.clicked.connect(self.register_user)
        layout.addWidget(self.register_btn)

        self.setLayout(layout)

    def register_user(self):
        username = self.new_username.text()
        password = self.new_password.text()
        if username and password:
            # Dodajemy użytkownika do tabeli (na sztywno, bez danych rejestracji)
            row_position = self.users_table.rowCount()
            self.users_table.insertRow(row_position)
            self.users_table.setItem(row_position, 0, QtWidgets.QTableWidgetItem(username))
            self.users_table.setItem(row_position, 1, QtWidgets.QTableWidgetItem("09:00 10.03.2025"))  # przykładowa data rejestracji
            self.users_table.setItem(row_position, 2, QtWidgets.QTableWidgetItem("00:00 10.03.2025"))  # przykładowa data ostatniego logowania
            self.users_table.setItem(row_position, 3, QtWidgets.QTableWidgetItem("Aktywne"))  # status

            QtWidgets.QMessageBox.information(self, "Rejestracja", f"Użytkownik {username} został zarejestrowany.")
        else:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Podaj nazwę użytkownika oraz hasło.")
