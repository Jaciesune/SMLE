import time
import mysql.connector
from fastapi import FastAPI, HTTPException

# Funkcja do czekania na bazę danych
def wait_for_db():
    while True:
        try:
            conn = mysql.connector.connect(
                host="mysql-db",
                port=3306,
                user="user",
                password="password",
                database="smle-database"
            )
            conn.close()
            print("✅ Baza danych jest dostępna!")
            break
        except mysql.connector.Error as err:
            print("⏳ Czekam na bazę danych...", err)
            time.sleep(5)

# Czekamy na dostępność bazy danych przed startem
wait_for_db()

# Inicjalizacja FastAPI
app = FastAPI()

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
@app.get("/users")
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Zmiana zapytania SQL, aby używać "last_login_date"
        cursor.execute("SELECT id, name, register_date, last_login_date, status FROM user")  
        users = cursor.fetchall()
    except mysql.connector.Error as err:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    cursor.close()
    conn.close()
    
    return users

# Endpoint dodający użytkownika
@app.post("/users")
def create_user(username: str, email: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO user (name, password, register_date, status) VALUES (%s, %s, NOW(), 'active')", (username, email))
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()
        cursor.close()
        conn.close()
        raise HTTPException(status_code=400, detail=f"Błąd: {err}")
    
    cursor.close()
    conn.close()
    
    return {"message": "Użytkownik dodany!"}

# Pętla podtrzymująca działanie aplikacji
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
