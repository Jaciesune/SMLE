import time
import mysql.connector
from fastapi import FastAPI, HTTPException
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
    dataset_dir: str
    epochs: int
    batch_size: int
    lr: float
    num_workers: int = 10
    patience: int = 8
    coco_gt_path: str = "/app/data/val/annotations/coco.json"
    num_augmentations: int = 8
    resume: str = None

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
        "--dataset_dir", request.dataset_dir,
        "--epochs", str(request.epochs),
        "--batch_size", str(request.batch_size),
        "--lr", str(request.lr),
        "--num_workers", str(request.num_workers),
        "--patience", str(request.patience),
        "--coco_gt_path", request.coco_gt_path,
        "--num_augmentations", str(request.num_augmentations),
    ]
    if request.resume:
        train_args.extend(["--resume", request.resume])

    result = detection_api.train_model(train_args)
    if "Błąd" in result:
        raise HTTPException(status_code=500, detail=result)
    return {"message": "Trening zakończony", "output": result}

# Inne endpointy
app.add_api_route("/users", get_users, methods=["GET"])
app.add_api_route("/users", create_user, methods=["POST"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)