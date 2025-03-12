from PyQt5 import QtWidgets

class UsersTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.users_list = QtWidgets.QListWidget()
        self.users_list.addItem("admin")
        self.users_list.addItem("user")
        layout.addWidget(self.users_list)

        form_layout = QtWidgets.QFormLayout()
        self.new_username = QtWidgets.QLineEdit()
        self.new_password = QtWidgets.QLineEdit()
        self.new_password.setEchoMode(QtWidgets.QLineEdit.Password)
        form_layout.addRow("Nazwa użytkownika:", self.new_username)
        form_layout.addRow("Hasło:", self.new_password)
        layout.addLayout(form_layout)

        self.register_btn = QtWidgets.QPushButton("Zarejestruj użytkownika")
        self.register_btn.clicked.connect(self.register_user)
        layout.addWidget(self.register_btn)

        self.setLayout(layout)

    def register_user(self):
        username = self.new_username.text()
        password = self.new_password.text()
        if username and password:
            self.users_list.addItem(username)
            QtWidgets.QMessageBox.information(self, "Rejestracja", f"Użytkownik {username} został zarejestrowany.")
        else:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Podaj nazwę użytkownika oraz hasło.")
