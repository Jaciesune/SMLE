import sys
import os
import logging

# Dodajemy katalog nadrzędny (SMLE) do sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5 import QtWidgets
from login_dialog import LoginDialog
from main_window import MainWindow
from utils import load_stylesheet  # Załadowanie funkcji z utils

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Określ ścieżki do plików stylów
    base_path = os.path.abspath(os.path.dirname(__file__))
    global_stylesheet_path = os.path.join(base_path, "frontend", "styles", "style.css")
    login_stylesheet_path = os.path.join(base_path, "frontend", "styles", "login_style.css")

    # Debugowanie: wypisz pełną ścieżkę do plików
    logger.debug(f"[DEBUG] Ścieżka do style.css: {global_stylesheet_path}")
    logger.debug(f"[DEBUG] Ścieżka do login_style.css: {login_stylesheet_path}")

    # Załaduj globalny styl
    global_stylesheet = load_stylesheet("frontend/styles/style.css")
    if not global_stylesheet:
        logger.error("[ERROR] Nie udało się wczytać style.css.")
    logger.debug("[DEBUG] Załadowano style.css")

    # Załaduj styl specyficzny dla logowania
    login_stylesheet = load_stylesheet("frontend/styles/login_style.css")
    if not login_stylesheet:
        logger.error("[ERROR] Nie udało się wczytać login_style.css.")
    logger.debug("[DEBUG] Załadowano login_style.css")

    # Połącz style (login_stylesheet nadpisze global_stylesheet w razie konfliktów) tylko dla logowania
    combined_stylesheet = global_stylesheet + "\n" + login_stylesheet
    app.setStyleSheet(combined_stylesheet)
    logger.debug("[DEBUG] Ustawiono style dla aplikacji")

    login = LoginDialog()
    logger.debug("[DEBUG] Pokazano okno logowania")
    if login.exec_() == QtWidgets.QDialog.Accepted:
        user_role = login.accepted_role
        user_name = login.accepted_username
        logger.debug(f"[DEBUG] Logowanie udane: user_role={user_role}, user_name={user_name}")
        try:
            main_window = MainWindow(user_role, user_name, combined_stylesheet)
            main_window.showFullScreen()
            logger.debug("[DEBUG] Uruchomiono MainWindow")
        except Exception as e:
            logger.error(f"[ERROR] Błąd podczas inicjalizacji MainWindow: {str(e)}")
            sys.exit(1)
    else:
        logger.debug("[DEBUG] Logowanie nieudane lub anulowane")
        sys.exit(0)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()