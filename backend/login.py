import mysql.connector
from fastapi import HTTPException

# Funkcja do pobierania użytkownika z bazy danych
def get_user(username):
    try:
        conn = mysql.connector.connect(
            host="mysql-db",
            port=3306,
            user="user",
            password="password",
            database="smle-database"
        )
        cur = conn.cursor()

        # Zmienione zapytanie do bazy - odwołujemy się teraz do tabeli 'user'
        cur.execute("SELECT name, password, status FROM user WHERE name = %s", (username,))
        user = cur.fetchone()

        conn.close()

        if user:
            return {"username": user[0], "password": user[1], "status": user[2]}
        return None
    except mysql.connector.Error as err:
        print(f"Błąd bazy danych: {err}")
        return None

# Funkcja do weryfikacji logowania
def verify_credentials(username: str, password: str):
    user = get_user(username)

    if not user:
        raise HTTPException(status_code=401, detail="Niepoprawne dane logowania")
    
    # Sprawdzamy, czy użytkownik ma aktywny status i czy hasło jest poprawne
    if user["password"] != password:
        raise HTTPException(status_code=401, detail="Niepoprawne dane logowania")
    
    if user["status"] != 'active':
        raise HTTPException(status_code=403, detail="Konto nieaktywne")

    # Możesz zwrócić rolę na podstawie nazwy użytkownika lub statusu
    return {"success": True, "role": "admin" if username == "admin" else "user"}
