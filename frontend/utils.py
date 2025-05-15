"""
Moduł zawierający funkcje narzędziowe dla aplikacji SMLE.

Dostarcza różne funkcje pomocnicze wykorzystywane w całej aplikacji,
w tym wczytywanie arkuszy stylów CSS oraz uwierzytelnianie użytkowników
poprzez komunikację z API backendu.
"""
#######################
# Importy bibliotek
#######################
import requests
import logging

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_stylesheet(filename):
    """
    Wczytuje zawartość pliku CSS z arkuszem stylów.
    
    Funkcja odczytuje plik CSS z katalogu styles i zwraca jego zawartość
    jako string, który może być bezpośrednio zastosowany do elementów Qt.
    
    Args:
        filename (str): Nazwa pliku CSS do wczytania (bez ścieżki)
        
    Returns:
        str: Zawartość pliku CSS lub pusty string w przypadku błędu
    """
    try:
        # Tworzymy pełną ścieżkę do pliku w katalogu frontend/styles/
        full_path = f"frontend/styles/{filename}"
        with open(full_path, "r", encoding="utf-8") as file:
            content = file.read()
            logger.debug(f"[DEBUG] Wczytano arkusz stylów z {full_path}, {len(content)} bajtów")
            return content
    except FileNotFoundError:
        logger.error(f"[ERROR] Plik stylów {full_path} nie został znaleziony.")
        return ""
    except Exception as e:
        logger.error(f"[ERROR] Błąd podczas wczytywania pliku stylów {full_path}: {e}")
        return ""

def verify_credentials(username, password, api_url):
    """
    Weryfikuje dane logowania użytkownika poprzez API backendu.
    
    Wysyła żądanie do endpointu /login z danymi uwierzytelniającymi
    i zwraca rolę użytkownika w przypadku pomyślnego logowania.
    
    Args:
        username (str): Nazwa użytkownika
        password (str): Hasło użytkownika
        api_url (str): Bazowy adres URL API backendu
        
    Returns:
        str or None: Rola użytkownika w przypadku pomyślnego uwierzytelnienia,
                     None w przypadku niepowodzenia
    """
    try:
        logger.debug(f"[DEBUG] Weryfikacja poświadczeń dla użytkownika: {username}")
        response = requests.post(
            f"{api_url}/login", 
            json={"username": username, "password": password}
        )
        logger.debug(f"[DEBUG] Status odpowiedzi: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.debug(f"[DEBUG] Odpowiedź JSON: {data}")
            
            if "role" in data:
                logger.info(f"[INFO] Użytkownik {username} zalogowany pomyślnie, rola: {data['role']}")
                return data["role"]
            else:
                logger.warning("[WARNING] Brak pola 'role' w odpowiedzi serwera.")
                return None
        else:
            logger.warning(f"[WARNING] Niepoprawne dane logowania: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"[ERROR] Błąd połączenia z serwerem: {e}")
        return None