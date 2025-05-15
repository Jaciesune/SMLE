"""
Główny moduł startowy aplikacji SMLE (System Maszynowego Liczenia Elementów).

Plik stanowi punkt wejściowy aplikacji, inicjalizuje interfejs użytkownika,
zarządza procesem uwierzytelniania oraz uruchamia główne okno programu
po pomyślnym zalogowaniu użytkownika.
"""

#######################
# Importy bibliotek
#######################
import sys
import os
import logging
from PyQt5 import QtWidgets, QtGui

#######################
# Dodanie katalogu nadrzędnego do ścieżki importu
#######################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#######################
# Importy lokalne
#######################
from login_dialog import LoginDialog
from main_window import MainWindow
from utils import load_stylesheet  # Załadowanie funkcji z utils

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """
    Główna funkcja uruchamiająca aplikację.
    
    Inicjalizuje aplikację PyQt, konfiguruje style i zasoby,
    uruchamia proces uwierzytelniania, a następnie tworzy
    i wyświetla główne okno aplikacji.
    """
    app = QtWidgets.QApplication(sys.argv)

    # Ustawienie nazwy aplikacji
    app.setApplicationName("SMLE")
    api_url = "http://localhost:8000"  

    # Ustawienie ikony dla całej aplikacji
    icon_path = "frontend/styles/images/icon.ico"
    icon = QtGui.QIcon(icon_path)
    if icon.isNull():
        logger.error(f"[ERROR] Nie udało się wczytać ikony z {icon_path}")
    else:
        app.setWindowIcon(icon)
        logger.debug(f"[DEBUG] Załadowano ikonę z {icon_path}")

    # Załaduj globalny styl
    global_stylesheet = load_stylesheet("style.css")
    if not global_stylesheet:
        logger.error("[ERROR] Nie udało się wczytać style.css")
        sys.exit(1)
    logger.debug("[DEBUG] Załadowano style.css")

    # Załaduj styl specyficzny dla logowania
    login_stylesheet = load_stylesheet("login_style.css")
    if not login_stylesheet:
        logger.error("[ERROR] Nie udało się wczytać login_style.css")
        sys.exit(1)
    logger.debug("[DEBUG] Załadowano login_style.css")

    # Połącz style (login_stylesheet nadpisze global_stylesheet w razie konfliktów)
    combined_stylesheet = global_stylesheet + "\n" + login_stylesheet
    app.setStyleSheet(combined_stylesheet)
    logger.debug("[DEBUG] Ustawiono style dla aplikacji")

    # Inicjalizacja okna logowania
    login = LoginDialog(api_url)
    logger.debug("[DEBUG] Pokazano okno logowania")
    
    # Uruchomienie okna logowania i obsługa wyniku
    if login.exec_() == QtWidgets.QDialog.Accepted:
        user_role = login.accepted_role
        user_name = login.accepted_username
        logger.debug(f"[DEBUG] Logowanie udane: user_role={user_role}, user_name={user_name}")
        try:
            main_window = MainWindow(user_role, user_name, combined_stylesheet, api_url)
            main_window.showFullScreen()
            logger.debug("[DEBUG] Uruchomiono MainWindow")
        except Exception as e:
            logger.error(f"[ERROR] Błąd podczas inicjalizacji MainWindow: {str(e)}")
            sys.exit(1)
    else:
        logger.debug("[DEBUG] Logowanie nieudane lub anulowane")
        sys.exit(0)

    # Uruchomienie pętli zdarzeń aplikacji
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()