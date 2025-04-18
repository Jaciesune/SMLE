import mysql.connector
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def get_db_connection():
    return mysql.connector.connect(
        host="mysql-db",
        port=3306,
        user="user",
        password="password",
        database="smle-database"
    )

class ModelPayload(BaseModel):
    name: str
    algorithm: str
    path: str
    epochs: int
    augmentations: int
    username: str

def get_user_id_by_username(username: str):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        logger.debug(f"Szukanie user_id dla username: {username}")
        cursor.execute("SELECT id FROM user WHERE name = %s", (username,))
        user = cursor.fetchone()
        logger.debug(f"Znaleziony użytkownik: {user}")
        if user is None:
            raise HTTPException(status_code=404, detail="Użytkownik nie znaleziony")
        return user['id']
    except mysql.connector.Error as err:
        logger.exception(f"Błąd zapytania do bazy danych przy pobieraniu user_id: {err}")
        raise HTTPException(status_code=500, detail=f"Błąd zapytania do bazy danych: {err}")
    finally:
        cursor.close()
        conn.close()

@router.post("/models/add")
def add_model(payload: ModelPayload):
    conn = None
    cursor = None

    try:
        logger.debug(f"Odebrany payload: {payload.dict()}")
        user_id = get_user_id_by_username(payload.username)
        logger.debug(f"Uzyskany user_id: {user_id}")

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        logger.debug("Wstawianie modelu do bazy danych...")
        cursor.execute("""
            INSERT INTO model (name, algorithm, version, creation_date, status, training_date, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            payload.name,
            payload.algorithm,
             "1.0",
            datetime.now(),
            "deployed",
            datetime.now(),
            user_id
        ))

        model_id = cursor.lastrowid  # <-- ZAPISUJEMY ID dodanego modelu

        logger.debug("Wstawianie wpisu do tabeli archive...")
        cursor.execute("""
            INSERT INTO archive (action, user_id, model_id, date)
            VALUES (%s, %s, %s, %s)
        """, (
            "Trenowanie",
            user_id,
            model_id,
            datetime.now()
        ))

        conn.commit()
        logger.info(f"Model {payload.name} został dodany do bazy i zarchiwizowany przez użytkownika {payload.username}.")
    except mysql.connector.Error as err:
        if conn:
            conn.rollback()
        logger.exception(f"Błąd przy dodawaniu modelu do bazy: {err}")
        raise HTTPException(status_code=500, detail=f"Błąd przy dodawaniu modelu: {err}")
    except Exception as ex:
        logger.exception(f"Niespodziewany wyjątek w add_model: {ex}")
        raise HTTPException(status_code=500, detail="Wystąpił nieoczekiwany błąd.")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return {"message": "Model zapisany i zarchiwizowany"}
