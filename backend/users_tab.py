import mysql.connector
from fastapi import HTTPException

# Funkcja do połączenia z bazą
def get_db_connection():
    return mysql.connector.connect(
        host="mysql-db",
        port=3306,
        user="user",
        password="password",
        database="smle-database"
    )

# Endpoint pobierający listę użytkowników
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Dodano kolumnę "role"
        cursor.execute("SELECT id, name, register_date, last_login_date, status, role FROM user")  
        users = cursor.fetchall()
    except mysql.connector.Error as err:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    cursor.close()
    conn.close()
    
    return users

# Endpoint dodający użytkownika
def create_user(data: dict):
    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "user")  # Domyślnie nowi użytkownicy dostają rolę "user"

    if not username or not password:
        raise HTTPException(status_code=400, detail="Nazwa użytkownika i hasło są wymagane")

    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO user (name, password, register_date, status, role) VALUES (%s, %s, NOW(), 'active', %s)", 
                       (username, password, role))
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()
        cursor.close()
        conn.close()
        raise HTTPException(status_code=400, detail=f"Błąd: {err}")
    
    cursor.close()
    conn.close()
    
    return {"message": "Użytkownik dodany!"}
