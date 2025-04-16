from PyQt5 import QtWidgets
from utils import verify_credentials

class LoginDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Logowanie")
        self.setFixedSize(350, 200)
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText("Nazwa użytkownika")
        layout.addWidget(self.username_input)

        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setPlaceholderText("Hasło")
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(self.password_input)

        self.login_btn = QtWidgets.QPushButton("Zaloguj")
        self.login_btn.clicked.connect(self.handle_login)
        layout.addWidget(self.login_btn)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet("color: red;")
        layout.addWidget(self.error_label)

        self.setLayout(layout)

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        role = verify_credentials(username, password)

        if role:
            print(f"Zalogowano jako {role}")  # Debugowanie roli
            self.accepted_role = role  # Przypisujemy rolę
            self.accepted_username = username
            self.accept()  # Akceptujemy logowanie i przechodzimy do głównego okna
        else:
            self.error_label.setText("Niepoprawne dane logowania!")
