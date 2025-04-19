import os
import glob
import re
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

        model_id = cursor.lastrowid

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

# Funkcja pomocnicza do szukania plików
def find_file_with_regex(model_name, timestamp_str, suffix, model_dir):
    # Budujemy wzorzec regex z użyciem daty, dowolnych znaków i końcówki
    pattern = f"{model_name}_{timestamp_str}.*{suffix}"
    logger.debug(f"Szukam pliku za pomocą wzorca: {pattern}")
    
    # Wyszukujemy pliki w katalogu, które pasują do wzorca
    matching_files = [f for f in os.listdir(model_dir) if re.match(pattern, f)]
    return matching_files

@router.delete("/models/{model_id}")
def delete_model(model_id: int):
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Pobieramy dane modelu wraz z datą utworzenia
        cursor.execute("SELECT name, algorithm, user_id, creation_date, status FROM model WHERE id = %s", (model_id,))
        model = cursor.fetchone()

        if not model:
            raise HTTPException(status_code=404, detail="Model nie znaleziony")
        
        if model['status'] == 'archived':
            raise HTTPException(status_code=400, detail="Model już jest zarchiwizowany i nie może być ponownie usunięty")

        model_name = model['name']
        algorithm = model['algorithm']
        user_id = model['user_id']
        creation_date = model['creation_date']

        model_dirs = {
            "MCNN": "/app/backend/MCNN/models",
            "MaskRCNN": "/app/backend/Mask_RCNN/models",
            "FasterRCNN": "/app/backend/FasterRCNN/models"
        }

        suffixes = {
            "MCNN": "_checkpoint.pth",
            "MaskRCNN": "_checkpoint.pth",
            "FasterRCNN": "_checkpoint.pth"
        }

        model_dir = model_dirs.get(algorithm)
        suffix = suffixes.get(algorithm)

        if not model_dir or not suffix:
            raise HTTPException(status_code=400, detail="Nieznany algorytm lub brak ścieżki")

        # Formatowanie daty jako timestamp do nazwy pliku (usuwamy sekundy)
        timestamp_str = creation_date.strftime("%Y%m%d_")  
        logger.debug(f"Szukam pliku z nazwą: {model_name}_{timestamp_str}*{suffix}")

        # Znajdowanie pliku za pomocą wyrażenia regularnego
        matching_files = find_file_with_regex(model_name, timestamp_str, suffix, model_dir)

        if matching_files:
            for file in matching_files:
                full_path = os.path.join(model_dir, file)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    logger.info(f"Usunięto plik modelu: {full_path}")
                else:
                    logger.warning(f"Nie znaleziono pliku do usunięcia: {full_path}")
        else:
            logger.warning(f"Nie znaleziono plików pasujących do wzorca: {model_name}_{timestamp_str}*{suffix}")

        # Archiwizacja w bazie
        cursor.execute("UPDATE model SET status = %s WHERE id = %s", ("archived", model_id))
        cursor.execute(""" 
            INSERT INTO archive (action, user_id, model_id, date)
            VALUES (%s, %s, %s, %s)
        """, (
            "Archiwizacja",
            user_id,
            model_id,
            datetime.now()
        ))

        conn.commit()
        logger.info(f"Model {model_name} oznaczony jako zarchiwizowany i plik usunięty.")
        return {"message": f"Model '{model_name}' został zarchiwizowany i plik usunięty."}

    except Exception as ex:
        if conn:
            conn.rollback()
        logger.exception(f"Błąd przy archiwizacji modelu: {ex}")
        raise HTTPException(status_code=500, detail=f"Błąd: {ex}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()