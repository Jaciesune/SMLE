"""
Implementacja funkcji backendu do zarządzania użytkownikami w aplikacji SMLE.

Moduł dostarcza funkcje do komunikacji z bazą danych MySQL, które umożliwiają
pobieranie listy użytkowników oraz tworzenie nowych kont użytkowników.
Jest wykorzystywany przez endpointy API związane z zarządzaniem użytkownikami.
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

def get_db_connection():
    """
    Tworzy i zwraca połączenie do bazy danych MySQL.
    
    Połączenie jest konfigurowane z parametrami serwera ustalonymi
    dla środowiska kontenerowego aplikacji.
    
    Returns:
        mysql.connector.connection: Obiekt połączenia z bazą danych
    
    Raises:
        mysql.connector.Error: W przypadku błędu połączenia z bazą danych
    """
    try:
        conn = mysql.connector.connect(
            host="mysql-db",
            port=3306,
            user="user",
            password="password",
            database="smle-database"
        )
        logger.debug("Utworzono połączenie z bazą danych dla modułu users_tab")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Błąd podczas łączenia z bazą danych: {err}")
        raise

def get_users():
    """
    Pobiera listę wszystkich użytkowników z bazy danych.
    
    Funkcja wykonuje zapytanie SQL pobierające dane wszystkich użytkowników
    z tabeli 'user', zawierające identyfikator, nazwę, datę rejestracji,
    datę ostatniego logowania, status oraz rolę.
    
    Returns:
        list: Lista słowników reprezentujących użytkowników systemu
        
    Raises:
        HTTPException: W przypadku błędu podczas pobierania danych
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT id, name, register_date, last_login_date, status, role FROM user")  
        users = cursor.fetchall()
        logger.info(f"Pobrano {len(users)} użytkowników z bazy danych")
        
        # Formatujemy dane użytkowników do odpowiedzi API
        formatted_users = []
        for user in users:
            formatted_users.append({
                "id": user["id"],
                "name": user["name"],
                "register_date": user["register_date"].strftime("%Y-%m-%d %H:%M:%S") if user["register_date"] else None,
                "last_login": user["last_login_date"].strftime("%Y-%m-%d %H:%M:%S") if user["last_login_date"] else None,
                "status": user["status"],
                "role": user["role"]
            })
        return formatted_users
        
    except mysql.connector.Error as err:
        logger.error(f"Błąd zapytania podczas pobierania użytkowników: {err}")
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    finally:
        cursor.close()
        conn.close()

def create_user(data: dict):
    """
    Dodaje nowego użytkownika do bazy danych.
    
    Funkcja tworzy nowy rekord użytkownika w tabeli 'user' z podanymi
    danymi, ustawiając domyślnie status 'active' oraz rolę 'user',
    jeśli nie podano innej roli.
    
    Args:
        data (dict): Słownik zawierający dane nowego użytkownika:
            - username (str): Nazwa użytkownika
            - password (str): Hasło użytkownika
            - role (str, optional): Rola użytkownika (domyślnie 'user')
        
    Returns:
        dict: Komunikat o powodzeniu operacji
        
    Raises:
        HTTPException: W przypadku braku wymaganych danych lub błędu bazy danych
    """
    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "user")  # Domyślnie nowi użytkownicy dostają rolę "user"

    if not username or not password:
        logger.warning("Próba utworzenia użytkownika bez podania nazwy lub hasła")
        raise HTTPException(status_code=400, detail="Nazwa użytkownika i hasło są wymagane")

    # Sprawdzanie, czy użytkownik o takiej nazwie już istnieje
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Najpierw sprawdzamy, czy użytkownik już istnieje
        cursor.execute("SELECT COUNT(*) FROM user WHERE name = %s", (username,))
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.warning(f"Próba utworzenia użytkownika o istniejącej nazwie: {username}")
            raise HTTPException(status_code=400, detail=f"Użytkownik o nazwie '{username}' już istnieje")
            
        # Dodajemy nowego użytkownika
        cursor.execute("""
            INSERT INTO user (name, password, register_date, status, role) 
            VALUES (%s, %s, NOW(), 'active', %s)
        """, (username, password, role))
        
        conn.commit()
        logger.info(f"Utworzono nowego użytkownika: {username} z rolą {role}")
        
    except mysql.connector.Error as err:
        conn.rollback()
        logger.error(f"Błąd bazy danych podczas tworzenia użytkownika {username}: {err}")
        raise HTTPException(status_code=400, detail=f"Błąd: {err}")
    
    finally:
        cursor.close()
        conn.close()
    
    return {"message": "Użytkownik dodany!", "username": username, "role": role}
