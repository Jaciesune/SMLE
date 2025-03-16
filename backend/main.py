import time
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from login import verify_credentials  # Importujemy funkcję z login.py
from users_tab import get_users, create_user

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
            print("✅ Baza danych jest dostępna! ✅")
            break
        except mysql.connector.Error as err:
            print("⏳ Oczekiwanie na bazę danych. ⏳", err)
            time.sleep(5)

# Czekamy na dostępność bazy danych przed startem
wait_for_db()

# Inicjalizacja FastAPI
app = FastAPI()

# Tworzymy model Pydantic do walidacji danych logowania
class LoginRequest(BaseModel):
    username: str
    password: str

# Endpoint logowania
@app.post("/login")
def login(request: LoginRequest):
    # Przekazujemy dane do funkcji verify_credentials
    role = verify_credentials(request.username, request.password)
    if role:
        return {"role": role}  # Zwracamy rolę użytkownika
    else:
        raise HTTPException(status_code=401, detail="Nieprawidłowe dane logowania")

# Inne endpointy
app.add_api_route("/users", get_users, methods=["GET"])
app.add_api_route("/users", create_user, methods=["POST"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
