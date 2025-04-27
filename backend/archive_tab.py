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

# Endpoint pobierający listę modeli
def get_archives():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Pobieramy dane modeli
        cursor.execute("SELECT id, action, user_id, model_id, date  FROM archive")
        archives = cursor.fetchall()
    except mysql.connector.Error as err:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    cursor.close()
    conn.close()
    
    return archives
