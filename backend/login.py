"""
Implementacja funkcji uwierzytelniania użytkowników w aplikacji SMLE.

Moduł dostarcza funkcje do weryfikacji poświadczeń użytkowników poprzez 
komunikację z bazą danych MySQL oraz zarządzanie procesem logowania.
"""
#######################
# Importy bibliotek
#######################
import mysql.connector
from fastapi import HTTPException
import logging

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user(username):
    """
    Pobiera dane użytkownika z bazy danych na podstawie nazwy użytkownika.
    
    Funkcja łączy się z bazą danych MySQL, wykonuje zapytanie o użytkownika
    o podanej nazwie i zwraca jego dane, jeśli użytkownik istnieje.
    
    Args:
        username (str): Nazwa użytkownika do wyszukania w bazie
        
    Returns:
        dict or None: Słownik z danymi użytkownika (username, password, status, role)
                     lub None, jeśli użytkownik nie został znaleziony
    """
    try:
        conn = mysql.connector.connect(
            host="mysql-db",
            port=3306,
            user="user",
            password="password",
            database="smle-database"
        )
        cur = conn.cursor()

        # Pobieramy dane użytkownika z bazy danych
        cur.execute("SELECT name, password, status, role FROM user WHERE name = %s", (username,))
        user = cur.fetchone()

        conn.close()

        if user:
            logger.info(f"Pobrano dane użytkownika: {username}")
            return {"username": user[0], "password": user[1], "status": user[2], "role": user[3]}
        
        logger.warning(f"Użytkownik nie znaleziony: {username}")
        return None
    except mysql.connector.Error as err:
        logger.error(f"Błąd bazy danych podczas pobierania użytkownika {username}: {err}")
        return None

def verify_credentials(username: str, password: str):
    """
    Weryfikuje dane uwierzytelniające użytkownika.
    
    Funkcja sprawdza, czy podane poświadczenia są poprawne poprzez
    porównanie z danymi w bazie danych oraz weryfikuje status konta.
    
    Args:
        username (str): Nazwa użytkownika
        password (str): Hasło użytkownika
        
    Returns:
        dict: Słownik zawierający status uwierzytelnienia, rolę i nazwę użytkownika
        
    Raises:
        HTTPException: W przypadku niepoprawnych danych logowania lub nieaktywnego konta
    """
    user = get_user(username)

    if not user:
        logger.warning(f"Nieudana próba logowania - nieznany użytkownik: {username}")
        raise HTTPException(status_code=401, detail="Niepoprawne dane logowania")
    
    # Sprawdzamy poprawność hasła
    if user["password"] != password:
        logger.warning(f"Nieudana próba logowania - błędne hasło dla użytkownika: {username}")
        raise HTTPException(status_code=401, detail="Niepoprawne dane logowania")
    
    # Sprawdzamy, czy konto jest aktywne
    if user["status"] != 'active':
        logger.warning(f"Nieudana próba logowania - nieaktywne konto: {username}")
        raise HTTPException(status_code=403, detail="Konto nieaktywne")

    logger.info(f"Pomyślne logowanie użytkownika: {username} z rolą {user['role']}")
    # Zwracamy dane użytkownika (rola i nazwa użytkownika)
    return {"success": True, "role": user["role"], "username": user["username"]}
