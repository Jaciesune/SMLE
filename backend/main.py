import sys
import os

from models_tab import get_db_connection, router as models_router
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from login import verify_credentials
from users_tab import get_users, create_user


from api.detection_api import DetectionAPI
from api.auto_label_api import AutoLabelAPI
from api.auto_label_routes import router as auto_label_router
from api.dataset_routes import router as dataset_router
from api.detection_routes import router as detection_router

from glob import glob
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def wait_for_db(max_attempts=12, wait_seconds=5):
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

if not wait_for_db():
    raise RuntimeError("Nie można uruchomić aplikacji bez połączenia z bazą danych.")

app = FastAPI()

detection_api = DetectionAPI()
auto_label_api = AutoLabelAPI()

# Rejestracja routera z auto_label_routes
app.include_router(auto_label_router)
app.include_router(dataset_router)
app.include_router(detection_router)
app.include_router(models_router)

class LoginRequest(BaseModel):
    username: str
    password: str

class DetectionRequest(BaseModel):
    image_path: str
    algorithm: str
    model_version: str

class TrainingRequest(BaseModel):
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

@app.post("/login")
def login(request: LoginRequest):
    auth_response = verify_credentials(request.username, request.password)
    if auth_response:
        # Zwracamy pełne dane, w tym username i role
        return {"role": auth_response["role"], "username": auth_response["username"]}
    else:
        raise HTTPException(status_code=401, detail="Nieprawidłowe dane logowania")

@app.get("/models")
def get_models():
    """Endpoint do pobierania listy modeli z bazy danych"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Zapytanie SQL, aby pobrać dane o modelach
        cursor.execute("SELECT id, name, algorithm, version, accuracy, creation_date, training_date, status FROM model")
        models = cursor.fetchall()  # Pobieramy wszystkie modele
    except mysql.connector.Error as err:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    cursor.close()
    conn.close()
    
    return models

@app.get("/archives")
def get_models():
    """Endpoint do pobierania listy archive z bazy danych"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Zapytanie SQL, aby pobrać dane o modelach
        cursor.execute("SELECT id, action, user_id, model_id, date  FROM archive")
        models = cursor.fetchall()  # Pobieramy wszystkie modele
    except mysql.connector.Error as err:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    
    cursor.close()
    conn.close()
    
    return models


@app.post("/train")
def train(request: TrainingRequest):
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
        return {"message": "Trening zakończony", "output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.add_api_route("/users", get_users, methods=["GET"])
app.add_api_route("/users", create_user, methods=["POST"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)