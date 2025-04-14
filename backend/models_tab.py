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
def get_models():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Pobieramy dane modeli
        cursor.execute("SELECT id, name, algorithm, version, creation_date, status, training_date FROM model")
        models = cursor.fetchall()
    except mysql.connector.Error as err:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    cursor.close()
    conn.close()
    
    return models
