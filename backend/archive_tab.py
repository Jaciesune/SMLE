"""
Implementacja funkcji backendu do obsługi archiwum zdarzeń w aplikacji SMLE.

Moduł dostarcza funkcje do komunikacji z bazą danych MySQL, które umożliwiają
pobieranie i zarządzanie historią zdarzeń w systemie, takich jak operacje
na modelach, działania użytkowników itp.
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
        logger.debug("Utworzono połączenie z bazą danych")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Błąd podczas łączenia z bazą danych: {err}")
        raise

def get_archives():
    """
    Pobiera listę wpisów z archiwum zdarzeń systemu.
    
    Funkcja wykonuje zapytanie SQL pobierające wszystkie wpisy
    z tabeli archive, zawierające informacje o wykonanych
    działaniach w systemie.
    
    Returns:
        list: Lista słowników reprezentujących wpisy archiwum
        
    Raises:
        HTTPException: W przypadku błędu podczas pobierania danych
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Pobieramy dane z archiwum
        cursor.execute("SELECT id, action, user_id, model_id, date FROM archive")
        archives = cursor.fetchall()
        logger.info(f"Pobrano {len(archives)} wpisów z archiwum")
    except mysql.connector.Error as err:
        logger.error(f"Błąd podczas wykonywania zapytania: {err}")
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    cursor.close()
    conn.close()
    
    return archives