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

        # Pobieramy także kolumnę 'role' z bazy danych
        cur.execute("SELECT name, password, status, role FROM user WHERE name = %s", (username,))
        user = cur.fetchone()

        conn.close()

        if user:
            return {"username": user[0], "password": user[1], "status": user[2], "role": user[3]}
        return None
    except mysql.connector.Error as err:
        print(f"Błąd bazy danych: {err}")
        return None

# Funkcja do weryfikacji logowania
def verify_credentials(username: str, password: str):
    user = get_user(username)

    if not user:
        raise HTTPException(status_code=401, detail="Niepoprawne dane logowania")
    
    # Sprawdzamy poprawność hasła
    if user["password"] != password:
        raise HTTPException(status_code=401, detail="Niepoprawne dane logowania")
    
    # Sprawdzamy, czy konto jest aktywne
    if user["status"] != 'active':
        raise HTTPException(status_code=403, detail="Konto nieaktywne")

    # Zwracamy rolę użytkownika z bazy danych
    return {"success": True, "role": user["role"]}
