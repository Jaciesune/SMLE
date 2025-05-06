import sys
import os
import json
from models_tab import get_db_connection, router as models_router
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
import mysql.connector
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from login import verify_credentials
from users_tab import get_users, create_user
from api.detection_api import DetectionAPI
from api.auto_label_api import AutoLabelAPI
from api.dataset_api import DatasetAPI
from api.auto_label_routes import router as auto_label_router
from api.detection_routes import router as detection_router
from api.dataset_routes import router as dataset_router
from api.benchmark_routes import router as benchmark_router

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detection_api = DetectionAPI()
auto_label_api = AutoLabelAPI()
dataset_api = DatasetAPI()

# Rejestracja routerów
app.include_router(auto_label_router)
app.include_router(dataset_router)
app.include_router(detection_router)
app.include_router(models_router)
app.include_router(benchmark_router)

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
        logger.debug(f"[DEBUG] Login successful: role={auth_response['role']}, username={auth_response['username']}")
        return {"role": auth_response["role"], "username": auth_response["username"]}
    else:
        logger.error("[DEBUG] Login failed: Invalid credentials")
        raise HTTPException(status_code=401, detail="Nieprawidłowe dane logowania")

@app.get("/models")
def get_models():
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
def get_models():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, action, user_id, model_id, date FROM archive")
        models = cursor.fetchall()
        logger.debug(f"[DEBUG] Pobrano archiwa: {models}")
    except mysql.connector.Error as err:
        conn.close()
        logger.error(f"[DEBUG] Błąd zapytania do archiwów: {err}")
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    finally:
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
        logger.debug(f"[DEBUG] Trening zakończony: {result}")
        return {"message": "Trening zakończony", "output": result}
    except Exception as e:
        logger.error(f"[DEBUG] Błąd podczas treningu: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.add_api_route("/users", get_users, methods=["GET"])
app.add_api_route("/users", create_user, methods=["POST"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)