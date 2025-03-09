import sys
from PyQt5 import QtWidgets
from login_dialog import LoginDialog
from main_window import MainWindow
from utils import load_stylesheet

def main():
    app = QtWidgets.QApplication(sys.argv)

    stylesheet = load_stylesheet("style.css")
    app.setStyleSheet(stylesheet)

    login = LoginDialog()
    if login.exec_() == QtWidgets.QDialog.Accepted:
        user_role = login.accepted_role
        main_window = MainWindow(user_role)
        main_window.showFullScreen()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
