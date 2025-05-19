"""
Implementacja funkcji backendu do zarządzania modelami w aplikacji SMLE.

Moduł dostarcza API do operacji na modelach uczenia maszynowego, takich jak
dodawanie nowych modeli, importowanie modeli z plików, usuwanie/archiwizacja
modeli oraz odczyt informacji o modelach z bazy danych MySQL.
"""
#######################
# Importy bibliotek
#######################
import os
import shutil
import re
import mysql.connector
from fastapi import HTTPException, APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from datetime import datetime
import logging

#######################
# Konfiguracja logowania
#######################
logger = logging.getLogger(__name__)
router = APIRouter()

def get_db_connection():
    """
    Tworzy i zwraca połączenie do bazy danych MySQL.
    
    Połączenie jest konfigurowane z parametrami serwera ustalonymi
    dla środowiska kontenerowego aplikacji.
    
    Returns:
        mysql.connector.connection: Obiekt połączenia z bazą danych
    """
    return mysql.connector.connect(
        host="mysql-db",
        port=3306,
        user="user",
        password="password",
        database="smle-database"
    )

def get_user_id_by_username(username: str):
    """
    Pobiera identyfikator użytkownika na podstawie jego nazwy użytkownika.
    
    Args:
        username (str): Nazwa użytkownika
        
    Returns:
        int: Identyfikator użytkownika w bazie danych
        
    Raises:
        HTTPException: Gdy użytkownik nie istnieje lub wystąpił błąd bazy danych
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        logger.debug(f"Szukanie user_id dla username: {username}")
        cursor.execute("SELECT id FROM user WHERE name = %s", (username,))
        user = cursor.fetchone()
        logger.debug(f"Znaleziony użytkownik: {user}")
        if user is None:
            raise HTTPException(status_code=400, detail="Użytkownik nie znaleziony")
        return user['id']
    except mysql.connector.Error as err:
        logger.exception(f"Błąd zapytania do bazy danych przy pobieraniu user_id: {err}")
        raise HTTPException(status_code=500, detail=f"Błąd zapytania do bazy danych: {err}")
    finally:
        cursor.close()
        conn.close()

class ModelPayload(BaseModel):
    """
    Model danych dla żądania dodania nowego modelu.
    
    Attributes:
        name (str): Nazwa modelu
        algorithm (str): Algorytm użyty w modelu
        path (str): Ścieżka do pliku modelu
        epochs (int): Liczba epok treningu
        augmentations (int): Liczba augmentacji użytych podczas treningu
        username (str): Nazwa użytkownika tworzącego model
    """
    name: str
    algorithm: str
    path: str
    epochs: int
    augmentations: int
    username: str

@router.post("/models/add")
def add_model(payload: ModelPayload):
    """
    Endpoint do dodawania nowego modelu do bazy danych.
    
    Zapisuje informacje o modelu w bazie danych oraz tworzy
    wpis w archiwum o operacji trenowania modelu.
    
    Args:
        payload (ModelPayload): Dane nowego modelu
        
    Returns:
        dict: Wiadomość o wyniku operacji
        
    Raises:
        HTTPException: W przypadku błędu podczas dodawania modelu
    """
    conn = None
    cursor = None

    try:
        logger.debug(f"Odebrany payload: {payload.dict()}")
        user_id = get_user_id_by_username(payload.username)
        logger.debug(f"Uzyskany user_id: {user_id}")

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Normalizacja algorytmu do zapisu w bazie
        normalized_algorithm = payload.algorithm.replace(" ", "-").replace("_", "-").upper()
        db_algorithm = payload.algorithm
        if normalized_algorithm == "MASK-RCNN":
            db_algorithm = "Mask R-CNN"
        elif normalized_algorithm == "FASTER-RCNN":
            db_algorithm = "Faster R-CNN"
        elif normalized_algorithm == "MCNN":
            db_algorithm = "MCNN"

        logger.debug("Wstawianie modelu do bazy danych...")
        cursor.execute(""" 
            INSERT INTO model (name, algorithm, version, creation_date, status, training_date, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            payload.name,
            db_algorithm,
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

def find_file_with_regex(model_name, timestamp_str, suffix, model_dir):
    """
    Wyszukuje pliki modelu na podstawie wzorca nazwy.
    
    Funkcja pomocnicza do odnajdywania plików modeli, które mogą mieć
    różne formaty nazw, ale zawierają nazwę modelu i określone rozszerzenie.
    
    Args:
        model_name (str): Podstawowa nazwa modelu
        timestamp_str (str): Znacznik czasu (nieużywany w aktualnej implementacji)
        suffix (str): Sufiks/rozszerzenie pliku
        model_dir (str): Katalog, w którym należy szukać plików
        
    Returns:
        list: Lista nazw plików pasujących do wzorca
    """
    # Pierwszy przypadek: dokładne dopasowanie do nazwy pliku
    exact_pattern = f"^{model_name}{suffix}$"
    
    # Drugi przypadek: dopasowanie z dowolnymi znakami pomiędzy
    wildcard_pattern = f"^{model_name}.*{suffix}$"

    logger.debug(f"Szukam pliku za pomocą wzorców: {exact_pattern} i {wildcard_pattern}")

    # Szukamy plików, które pasują do któregokolwiek z wzorców
    matching_files = [
        f for f in os.listdir(model_dir)
        if re.match(exact_pattern, f) or re.match(wildcard_pattern, f)
    ]
    return matching_files

@router.post("/models/upload")
def upload_model(
    algorithm: str = Form(...),
    file: UploadFile = File(...),
    user_name: str = Form(...)
):
    """
    Endpoint do wczytywania nowego modelu z pliku.
    
    Przyjmuje plik modelu (.pth), zapisuje go w odpowiednim katalogu
    w zależności od algorytmu oraz dodaje informacje o modelu do bazy danych.
    
    Args:
        algorithm (str): Algorytm modelu (Mask R-CNN, FasterRCNN, MCNN)
        file (UploadFile): Plik modelu w formacie .pth
        user_name (str): Nazwa użytkownika wczytującego model
        
    Returns:
        dict: Wiadomość o wyniku operacji
        
    Raises:
        HTTPException: W przypadku błędu podczas wczytywania modelu
    """
    try:
        logger.debug(f"Przesłana wartość algorithm: '{algorithm}'")
        if not file.filename.endswith(".pth"):
            raise HTTPException(status_code=400, detail="Plik musi mieć rozszerzenie .pth")

        # Wyciągamy nazwę modelu do pierwszego _
        model_name = os.path.splitext(file.filename)[0].split('_')[0]

        # Mapowanie algorytmów na ścieżki folderów
        target_dirs = {
            "MCNN": "/app/backend/MCNN/models",
            "MASK-RCNN": "/app/backend/Mask_RCNN/models",
            "FASTER-RCNN": "/app/backend/FasterRCNN/saved_models"
        }

        # Normalizacja przesłanego algorytmu do klucza
        normalized_algorithm = algorithm.replace(" ", "-").replace("_", "-").upper()
        logger.debug(f"Normalizowana wartość algorithm do mapowania: '{normalized_algorithm}'")

        matched_dir = target_dirs.get(normalized_algorithm)
        if not matched_dir:
            raise HTTPException(
                status_code=400,
                detail=f"Nieobsługiwany algorytm: '{algorithm}'. Oczekiwane wartości: MCNN, MASK-RCNN, FASTER-RCNN"
            )

        os.makedirs(matched_dir, exist_ok=True)
        logger.debug(f"Ścieżka docelowa dla modelu: {matched_dir}")

        save_path = os.path.join(matched_dir, file.filename)
        logger.debug(f"Zapisywanie modelu w: {save_path}")

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Zapisz dane do bazy
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        user_id = get_user_id_by_username(user_name)

        now = datetime.now()
        version = "1.0"
        status = "deployed"

        # Normalizacja algorytmu do zapisu w bazie (Mask R-CNN)
        db_algorithm = algorithm
        if normalized_algorithm == "MASK-RCNN":
            db_algorithm = "Mask R-CNN"
        elif normalized_algorithm == "FASTER-RCNN":
            db_algorithm = "Faster R-CNN"
        elif normalized_algorithm == "MCNN":
            db_algorithm = "MCNN"

        cursor.execute(""" 
            INSERT INTO model (name, algorithm, version, creation_date, status, training_date, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            model_name,
            db_algorithm,
            version,
            now,
            status,
            now,
            user_id
        ))

        model_id = cursor.lastrowid

        cursor.execute(""" 
            INSERT INTO archive (action, user_id, model_id, date)
            VALUES (%s, %s, %s, %s)
        """, (
            "Import",
            user_id,
            model_id,
            now
        ))

        conn.commit()
        cursor.close()
        conn.close()

        return {"message": f"Model '{file.filename}' został zaimportowany i zapisany do bazy danych."}

    except Exception as e:
        logger.exception(f"Błąd podczas uploadu modelu: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd podczas uploadu modelu: {e}")

@router.delete("/models/{model_id}")
def delete_model(model_id: int):
    """
    Endpoint do usuwania/archiwizacji modelu.
    
    Oznacza model jako zarchiwizowany w bazie danych oraz
    fizycznie usuwa plik modelu z systemu plików, jeśli istnieje.
    
    Args:
        model_id (int): Identyfikator modelu do usunięcia
        
    Returns:
        dict: Wiadomość o wyniku operacji
        
    Raises:
        HTTPException: W przypadku błędu podczas usuwania modelu
    """
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT name, algorithm, user_id, creation_date, status FROM model WHERE id = %s", (model_id,))
        model = cursor.fetchone()

        if not model:
            raise HTTPException(status_code=404, detail="Model nie znaleziony")
        
        if model['status'] == 'archived':
            logger.warning(f"Próba usunięcia zarchiwizowanego modelu o ID {model_id} (nazwa: {model['name']})")
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model['name']}' (ID: {model_id}) jest już zarchiwizowany i nie może być ponownie usunięty."
            )

        model_name = model['name']
        algorithm = model['algorithm']
        user_id = model['user_id']

        # Mapowanie algorytmów z bazy na foldery
        model_dirs = {
            "MCNN": "/app/backend/MCNN/models",
            "Mask R-CNN": "/app/backend/Mask_RCNN/models",
            "Faster R-CNN": "/app/backend/FasterRCNN/saved_models"
        }

        suffixes = {
            "MCNN": "_checkpoint.pth",
            "Mask R-CNN": "_checkpoint.pth",
            "Faster R-CNN": "_checkpoint.pth"
        }

        # Normalizacja algorytmu z bazy do klucza w model_dirs
        normalized_algorithm = algorithm.replace(" ", "-").replace("_", "-").upper()
        logger.debug(f"Normalizowana wartość algorithm z bazy: '{normalized_algorithm}'")

        # Mapowanie na odpowiedni klucz w model_dirs
        mapped_algorithm = {
            "MASK-RCNN": "Mask R-CNN",
            "FASTER-RCNN": "Faster R-CNN",
            "MCNN": "MCNN"
        }.get(normalized_algorithm, algorithm)

        model_dir = model_dirs.get(mapped_algorithm)
        suffix = suffixes.get(mapped_algorithm)

        if not model_dir or not suffix:
            raise HTTPException(status_code=400, detail=f"Nieznany algorytm lub brak ścieżki dla algorithm: {algorithm}")

        logger.debug(f"Szukam pliku z nazwą: {model_name}{suffix}")

        matching_files = find_file_with_regex(model_name, "", suffix, model_dir)

        if matching_files:
            for file in matching_files:
                full_path = os.path.join(model_dir, file)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    logger.info(f"Usunięto plik modelu: {full_path}")
                else:
                    logger.warning(f"Nie znaleziono pliku do usunięcia: {full_path}")
        else:
            logger.warning(f"Nie znaleziono plików pasujących do wzorca: {model_name}{suffix}")

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
        return {"message": f"Model '{model_name}' (ID: {model_id}) został zarchiwizowany i plik usunięty."}

    except HTTPException as http_exc:
        logger.error(f"Wystąpił kontrolowany błąd HTTP: {http_exc.detail}")
        raise http_exc
    except Exception as ex:
        if conn:
            conn.rollback()
        logger.exception(f"Błąd przy archiwizacji modelu: {ex}")
        raise HTTPException(status_code=500, detail=f"Wystąpił błąd serwera: {str(ex)}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()