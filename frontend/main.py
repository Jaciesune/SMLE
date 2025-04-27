import sys
import os
# Dodajemy katalog nadrzędny (SMLE) do sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5 import QtWidgets
from login_dialog import LoginDialog
from main_window import MainWindow
from utils import load_stylesheet  # Załadowanie funkcji z utils

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Załaduj CSS z pliku style.css znajdującego się w folderze frontend
    stylesheet = load_stylesheet("frontend/style.css")  # Upewnij się, że ścieżka jest poprawna
    app.setStyleSheet(stylesheet)  # Ustawienie CSS w aplikacji

    login = LoginDialog()
    if login.exec_() == QtWidgets.QDialog.Accepted:
        user_role = login.accepted_role
        user_name = login.accepted_username
        main_window = MainWindow(user_role, user_name)
        main_window.showFullScreen()


    sys.exit(app.exec_())

if __name__ == "__main__":
    main()