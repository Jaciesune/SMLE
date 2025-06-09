"""
Główny moduł backendu aplikacji SMLE (System Maszynowego Liczenia Elementów).

Moduł inicjalizuje serwer FastAPI, konfiguruje połączenie z bazą danych MySQL,
rejestruje routery poszczególnych komponentów oraz definiuje podstawowe
endpointy API. Stanowi centralny punkt wejścia dla całego backendu aplikacji.
"""
#######################
# Importy bibliotek
#######################
import sys
import os
import json
import time
import mysql.connector
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#######################
# Dodanie katalogu nadrzędnego do ścieżki importu
#######################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

#######################
# Importy lokalne
#######################
from models_tab import get_db_connection, router as models_router
from login import verify_credentials
from users_tab import get_users, create_user, update_user, update_user_status

#######################
# Importy API i routerów
#######################
from api.detection_api import DetectionAPI
from api.auto_label_api import AutoLabelAPI
from api.dataset_api import DatasetAPI
from api.auto_label_routes import router as auto_label_router
from api.detection_routes import router as detection_router
from api.dataset_routes import router as dataset_router
from api.benchmark_routes import router as benchmark_router

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def wait_for_db(max_attempts=12, wait_seconds=5):
    """
    Czeka na dostępność bazy danych, wykonując próby połączenia.
    
    Funkcja próbuje połączyć się z bazą danych MySQL w określonych odstępach
    czasu, aż osiągnie maksymalną liczbę prób. Jest używana podczas uruchamiania
    aplikacji w środowisku kontenerowym, gdzie baza danych może potrzebować
    więcej czasu na inicjalizację niż sam serwer API.
    
    Args:
        max_attempts (int, optional): Maksymalna liczba prób połączenia
        wait_seconds (int, optional): Czas oczekiwania między próbami w sekundach
        
    Returns:
        bool: True jeśli połączenie zostało ustanowione, False w przeciwnym razie
    """
    attempts = 0
    while attempts < max_attempts:
        try:
            conn = mysql.connector.connect(
                host="mysql-db",
                port=3306,
                user="user",
                password="password",
                database="smle-database"
            )
            conn.close()
            logger.info("✅ Baza danych jest dostępna! ✅")
            return True
        except mysql.connector.Error as err:
            attempts += 1
            logger.error("⏳ Oczekiwanie na bazę danych, próba %d/%d: %s", attempts, max_attempts, err)
            time.sleep(wait_seconds)
    logger.error("❌ Nie udało się połączyć z bazą danych po %d próbach.", max_attempts)
    return False

# Sprawdzenie dostępności bazy danych przed uruchomieniem serwera
if not wait_for_db():
    raise RuntimeError("Nie można uruchomić aplikacji bez połączenia z bazą danych.")

# Inicjalizacja aplikacji FastAPI
app = FastAPI()

# Konfiguracja CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicjalizacja kluczowych komponentów API
detection_api = DetectionAPI()
auto_label_api = AutoLabelAPI()
dataset_api = DatasetAPI()

# Rejestracja routerów dla poszczególnych modułów funkcjonalnych
app.include_router(auto_label_router)
app.include_router(dataset_router)
app.include_router(detection_router)
app.include_router(models_router)
app.include_router(benchmark_router)

# Modele danych dla żądań API
class LoginRequest(BaseModel):
    """
    Model danych dla żądania logowania.
    
    Attributes:
        username (str): Nazwa użytkownika
        password (str): Hasło użytkownika
    """
    username: str
    password: str

class DetectionRequest(BaseModel):
    """
    Model danych dla żądania detekcji obiektów na obrazie.
    
    Attributes:
        image_path (str): Ścieżka do obrazu
        algorithm (str): Nazwa algorytmu do wykorzystania
        model_version (str): Wersja modelu
    """
    image_path: str
    algorithm: str
    model_version: str

class TrainingRequest(BaseModel):
    """
    Model danych dla żądania treningu modelu.
    
    Zawiera wszystkie parametry potrzebne do skonfigurowania 
    procesu treningu modelu uczenia maszynowego.
    
    Attributes:
        train_dir (str): Katalog z danymi treningowymi
        epochs (int): Liczba epok treningu
        lr (float): Współczynnik uczenia
        model_name (str): Nazwa modelu
        coco_train_path (str): Ścieżka do adnotacji treningowych w formacie COCO
        coco_gt_path (str): Ścieżka do adnotacji walidacyjnych w formacie COCO
        host_train_path (str): Ścieżka do danych treningowych na hoście
        host_val_path (str): Ścieżka do danych walidacyjnych na hoście
        num_augmentations (int): Liczba augmentacji na obraz
        resume (str): Nazwa modelu do wznowienia treningu (lub None)
        batch_size (int): Rozmiar batcha
        num_workers (int): Liczba równoległych wątków ładowania danych
        patience (int): Parametr cierpliwości dla early stopping
    """
    train_dir: str
    epochs: int
    lr: float
    model_name: str
    coco_train_path: str
    coco_gt_path: str = "/app/backend/data/val/annotations/instances_val.json"
    host_train_path: str
    host_val_path: str = None
    num_augmentations: int = 8
    resume: str = None
    batch_size: int = 4
    num_workers: int = 10
    patience: int = 8

class UserUpdate(BaseModel):
    username: str
    password: str = None

class UserStatusUpdate(BaseModel):
    status: str

@app.post("/login")
def login(request: LoginRequest):
    """
    Endpoint uwierzytelniania użytkowników.
    
    Weryfikuje dane logowania i zwraca informacje o użytkowniku
    w przypadku pomyślnego uwierzytelnienia.
    
    Args:
        request (LoginRequest): Dane uwierzytelniające
        
    Returns:
        dict: Informacje o zalogowanym użytkowniku (rola i nazwa)
        
    Raises:
        HTTPException: W przypadku niepoprawnych danych logowania
    """
    auth_response = verify_credentials(request.username, request.password)
    if auth_response:
        logger.debug(f"[DEBUG] Login successful: role={auth_response['role']}, username={auth_response['username']}")
        return {"role": auth_response["role"], "username": auth_response["username"]}
    else:
        logger.error("[DEBUG] Login failed: Invalid credentials")
        raise HTTPException(status_code=401, detail="Nieprawidłowe dane logowania")

@app.get("/models")
def get_models():
    """
    Pobiera listę wszystkich modeli z bazy danych.
    
    Endpoint zwraca szczegółowe informacje o wszystkich dostępnych 
    modelach w systemie, formatując odpowiedź z dodatkowymi metadanymi.
    
    Returns:
        list: Lista słowników z informacjami o modelach
        
    Raises:
        HTTPException: W przypadku błędu zapytania do bazy danych
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, name, algorithm, version, accuracy, creation_date, training_date, status FROM model")
        models = cursor.fetchall()
        logger.debug(f"[DEBUG] Pobrano modele: {models}")
        # Formatowanie odpowiedzi z użyciem nazwy modelu
        formatted_models = [
            {
                "id": m["id"],
                "name": m["name"],
                "algorithm": m["algorithm"],
                "version": m["version"],
                "accuracy": m["accuracy"],
                "creation_date": m["creation_date"],
                "training_date": m["training_date"],
                "status": m["status"],
                "display_name": f"{m['name']} ({m['algorithm']} - v{m['version']})"
            }
            for m in models
        ]
        return formatted_models
    except mysql.connector.Error as err:
        conn.close()
        logger.error(f"[DEBUG] Błąd zapytania do bazy modeli: {err}")
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    finally:
        cursor.close()
        conn.close()

@app.get("/archives")
def get_archives():
    """
    Pobiera listę wszystkich wpisów z archiwum zdarzeń.
    
    Endpoint zwraca historię działań wykonanych w systemie,
    takich jak operacje na modelach czy działania użytkowników.
    Zwraca nazwę użytkownika zamiast ID oraz nazwę modelu w formacie 'algorithm - name'.
    
    Returns:
        list: Lista słowników z wpisami archiwum
        
    Raises:
        HTTPException: W przypadku błędu zapytania do bazy danych
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT archive.id, archive.action, archive.user_id, archive.model_id, 
                   archive.date, user.name AS username, model.name AS model_name, 
                   model.algorithm
            FROM archive
            LEFT JOIN user ON archive.user_id = user.id
            LEFT JOIN model ON archive.model_id = model.id
        """)
        archives = cursor.fetchall()
        logger.debug(f"[DEBUG] Pobrano archiwa: {archives}")
        # Formatowanie odpowiedzi
        formatted_archives = [
            {
                "id": a["id"],
                "action": a["action"],
                "model_display_name": f"{a['algorithm']} - {a['model_name']}" if a["model_name"] and a["algorithm"] else "Nieznany model",
                "username": a["username"] or "Nieznany",
                "date": a["date"].strftime("%Y-%m-%d %H:%M:%S") if a["date"] else None
            }
            for a in archives
        ]
        return formatted_archives
    except mysql.connector.Error as err:
        conn.close()
        logger.error(f"[DEBUG] Błąd zapytania do archiwów: {err}")
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    finally:
        cursor.close()
        conn.close()

@app.post("/train")
def train(request: TrainingRequest):
    """
    Inicjuje proces trenowania modelu.
    
    Endpoint przetwarza żądanie treningu modelu, przygotowuje argumenty
    dla procesu trenowania i uruchamia trening poprzez API detekcji.
    
    Args:
        request (TrainingRequest): Parametry konfiguracyjne treningu
        
    Returns:
        dict: Wiadomość o wyniku operacji trenowania
        
    Raises:
        HTTPException: W przypadku błędu podczas treningu
    """
    train_args = [
        "--train_dir", request.train_dir,
        "--epochs", str(request.epochs),
        "--lr", str(request.lr),
        "--model_name", request.model_name,
        "--coco_train_path", request.coco_train_path,
        "--coco_gt_path", request.coco_gt_path,
        "--host_train_path", request.host_train_path,
        "--num_augmentations", str(request.num_augmentations),
        "--batch_size", str(request.batch_size),
        "--num_workers", str(request.num_workers),
        "--patience", str(request.patience),
    ]
    if request.host_val_path:
        train_args.extend(["--host_val_path", request.host_val_path])
    if request.resume:
        train_args.extend(["--resume", request.resume])
    try:
        result = detection_api.train_model(train_args)
        logger.debug(f"[DEBUG] Trening zakończony: {result}")
        return {"message": "Trening zakończony", "output": result}
    except Exception as e:
        logger.error(f"[DEBUG] Błąd podczas treningu: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dodanie endpointów dla zarządzania użytkownikami
app.add_api_route("/users", get_users, methods=["GET"])
app.add_api_route("/users", create_user, methods=["POST"])

@app.put("/users/{user_id}")
def update_user_endpoint(user_id: int, request: UserUpdate):
    """
    Aktualizuje dane użytkownika.
    
    Args:
        user_id (int): ID użytkownika
        request (UserUpdate): Nowe dane użytkownika
        
    Returns:
        dict: Komunikat o powodzeniu operacji
    """
    return update_user(user_id, request)

@app.put("/users/{user_id}/status")
def update_user_status_endpoint(user_id: int, request: UserStatusUpdate):
    """
    Aktualizuje status użytkownika.
    
    Args:
        user_id (int): ID użytkownika
        request (UserStatusUpdate): Nowy status
        
    Returns:
        dict: Komunikat o powodzeniu operacji
    """
    return update_user_status(user_id, request)


# Uruchomienie serwera
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)