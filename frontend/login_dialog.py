from PyQt5 import QtWidgets, QtGui, QtCore
import logging

from utils import verify_credentials  

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LoginDialog(QtWidgets.QDialog):
    def __init__(self, api_url):
        super().__init__()
        self.setWindowTitle("Logowanie")
        self.setObjectName("LoginDialog")
        self.api_url = api_url
        self.setFixedSize(500, 350)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setWindowIcon(QtGui.QIcon("frontend/styles/images/icon.ico"))
        # Włączenie przezroczystości tła dla zaokrąglonych rogów
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        logger.debug("[DEBUG] Inicjalizacja LoginDialog")
        self.init_ui()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # Zerowe marginesy dla całego okna
        main_layout.setSpacing(0)  # Bez odstępów między elementami

        # Pasek tytułowy
        title_bar = self.init_title_bar()
        main_layout.addWidget(title_bar)

        # Kontener dla treści panelu logowania
        content_widget = QtWidgets.QWidget()
        content_widget.setObjectName("ContentWidget")  # Dodajemy unikalny identyfikator
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 20)  # Większy górny margines (40px)
        content_layout.setSpacing(30)  # Większy odstęp między elementami (30px)
        content_layout.setAlignment(QtCore.Qt.AlignTop)

        # Pola tekstowe
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText("Nazwa użytkownika")
        content_layout.addWidget(self.username_input)

        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setPlaceholderText("Hasło")
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        content_layout.addWidget(self.password_input)

        # Przycisk logowania
        self.login_btn = QtWidgets.QPushButton("Zaloguj")
        self.login_btn.setDefault(True)  # Ustawienie jako domyślny przycisk
        self.login_btn.setAutoDefault(True)  # Enter w QLineEdit aktywuje przycisk
        self.login_btn.setStyleSheet("background-color: #263859")
        self.login_btn.clicked.connect(self.handle_login)
        content_layout.addWidget(self.login_btn)

        # Etykieta błędu
        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet("color: red;")
        content_layout.addWidget(self.error_label)

        content_layout.addStretch()  # Rozciągnięcie, aby elementy były u góry

        main_layout.addWidget(content_widget)
        self.setLayout(main_layout)

        # Ustawienie fokusu na polu nazwy użytkownika
        self.username_input.setFocus()

    def init_title_bar(self):
        title_bar = QtWidgets.QFrame(self)
        title_bar.setObjectName("TitleBar")
        title_bar_layout = QtWidgets.QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)  # Zerowe marginesy
        title_bar_layout.setSpacing(0)  # Zerowe odstępy

        title_label = QtWidgets.QLabel("Logowanie")
        title_label.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold; margin-left: 20px; padding: 5px; background-color: #1C2228;")
        title_bar_layout.addWidget(title_label)

        title_bar_layout.addStretch()  # Rozciągnięcie dla wyrównania przycisku do prawej krawędzi

        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setObjectName("CloseButton")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.close)
        title_bar_layout.addWidget(close_btn)

        return title_bar

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        logger.debug(f"[DEBUG] Próba logowania: username={username}")

        try:
            role = verify_credentials(username, password, self.api_url)
            logger.debug(f"[DEBUG] Wynik verify_credentials: role={role}")

            if role:
                logger.debug(f"[DEBUG] Logowanie udane: role={role}")
                self.accepted_role = role
                self.accepted_username = username
                self.accept()
            else:
                logger.debug("[DEBUG] Logowanie nieudane: nieprawidłowe dane")
                self.error_label.setText("Niepoprawne dane logowania!")
        except Exception as e:
            logger.error(f"[ERROR] Błąd podczas logowania: {str(e)}")
            self.error_label.setText("Błąd logowania: spróbuj ponownie")

    def keyPressEvent(self, event):
        # Przechwytujemy Enter, aby nie wywoływał domyślnych akcji QDialog
        if event.key() in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return):
            self.login_btn.click()  # Wywołanie akcji przycisku Zaloguj
            event.accept()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.childAt(event.pos()) in [self.findChild(QtWidgets.QFrame, "TitleBar")]:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'drag_position'):
            if event.buttons() & QtCore.Qt.LeftButton:
                self.move(event.globalPos() - self.drag_position)
                event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and hasattr(self, 'drag_position'):
            delattr(self, 'drag_position')