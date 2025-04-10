import time
import mysql.connector
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from login import verify_credentials
from users_tab import get_users, create_user
from api.detection_api import DetectionAPI

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

# Inicjalizacja DetectionAPI
detection_api = DetectionAPI()

# Tworzymy model Pydantic do walidacji danych logowania
class LoginRequest(BaseModel):
    username: str
    password: str

# Model Pydantic do żądania detekcji
class DetectionRequest(BaseModel):
    image_path: str
    algorithm: str
    model_version: str

# Model Pydantic do żądania treningu
class TrainingRequest(BaseModel):
    train_dir: str  # Zmieniono z dataset_dir na train_dir dla zgodności
    epochs: int
    lr: float
    model_name: str  # Dodano
    coco_train_path: str  # Dodano
    coco_gt_path: str = "/app/backend/data/val/annotations/instances_val.json"
    host_train_path: str  # Dodano
    host_val_path: str = None  # Dodano (opcjonalne)
    num_augmentations: int = 8
    resume: str = None
    batch_size: int = 4  # Zachowano, jeśli skrypty treningowe tego wymagają
    num_workers: int = 10  # Zachowano
    patience: int = 8  # Zachowano

# Endpoint logowania
@app.post("/login")
def login(request: LoginRequest):
    auth_response = verify_credentials(request.username, request.password)
    if auth_response:
        return {"role": auth_response["role"]}
    else:
        raise HTTPException(status_code=401, detail="Nieprawidłowe dane logowania")

# Endpoint do detekcji
@app.post("/detect")
def detect(request: DetectionRequest):
    result = detection_api.analyze_with_model(
        request.image_path, request.algorithm, request.model_version
    )
    if "Błąd" in result:
        raise HTTPException(status_code=500, detail=result)
    return {"result_path": result}

# Endpoint do treningu
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

# Inne endpointy
app.add_api_route("/users", get_users, methods=["GET"])
app.add_api_route("/users", create_user, methods=["POST"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)