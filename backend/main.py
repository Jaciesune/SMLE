import sys
import os
import shutil
import json
from models_tab import get_db_connection, router as models_router
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
import mysql.connector
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from login import verify_credentials
from users_tab import get_users, create_user
from api.detection_api import DetectionAPI
from api.auto_label_api import AutoLabelAPI
from api.auto_label_routes import router as auto_label_router
from api.dataset_routes import router as dataset_router
from api.detection_routes import router as detection_router
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

class BenchmarkRequest(BaseModel):
    algorithm: str
    model_version: str
    model_name: str  # Dodajemy model_name
    image_folder: str
    annotation_path: str

class ImageDataset:
    def __init__(self, image_folder, annotation_path):
        self.image_folder = image_folder
        self.annotation_path = annotation_path
        logger.debug(f"[DEBUG] Inicjalizacja ImageDataset: image_folder={self.image_folder}, annotation_path={self.annotation_path}")
        
        # Wczytaj listę obrazów
        self.images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
        logger.debug(f"[DEBUG] Znalezione obrazy: {self.images}")
        if not self.images:
            logger.warning(f"[DEBUG] Brak obrazów w folderze {self.image_folder}")

        # Wczytaj annotacje (format LabelMe)
        self.file_to_annotations = {}
        for img_file in self.images:
            annotation_file = os.path.join(self.annotation_path, f"{os.path.splitext(img_file)[0]}.json")
            logger.debug(f"[DEBUG] Szukam annotacji dla {img_file}: {annotation_file}")
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    ann_data = json.load(f)
                self.file_to_annotations[img_file] = ann_data.get("shapes", [])
                logger.debug(f"[DEBUG] Znaleziono annotacje dla {img_file}: {len(self.file_to_annotations[img_file])} kształtów")
            else:
                logger.warning(f"[DEBUG] Brak pliku annotacji dla obrazu {img_file}")
                self.file_to_annotations[img_file] = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        img_path = os.path.join(self.image_folder, file_name)
        annotations = self.file_to_annotations.get(file_name, [])
        return img_path, annotations

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
    except mysql.connector.Error as err:
        conn.close()
        logger.error(f"[DEBUG] Błąd zapytania do bazy modeli: {err}")
        raise HTTPException(status_code=500, detail=f"Błąd zapytania: {err}")
    finally:
        cursor.close()
        conn.close()
    return models

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

@app.post("/prepare_benchmark_data")
async def prepare_benchmark_data(
    images: list[UploadFile] = File(...),
    annotations: list[UploadFile] = File(...),
    request: Request = None
):
    user_role = request.headers.get("X-User-Role")
    logger.debug(f"[DEBUG] Wywołanie /prepare_benchmark_data: user_role={user_role}, liczba obrazów={len(images)}")
    if user_role != "admin":
        logger.error("[DEBUG] Brak uprawnień: użytkownik nie jest adminem")
        raise HTTPException(status_code=403, detail="Only admins can prepare benchmark data")

    if len(images) != len(annotations):
        logger.error(f"[DEBUG] Niezgodna liczba obrazów ({len(images)}) i annotacji ({len(annotations)})")
        raise HTTPException(status_code=400, detail="Number of images and annotations must match")

    # Ścieżki w kontenerze
    image_folder = "/app/backend/data/test/images"
    annotation_path = "/app/backend/data/test/annotations"

    # Wyczyść stare dane
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    if os.path.exists(annotation_path):
        shutil.rmtree(annotation_path)

    # Stwórz katalogi
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(annotation_path, exist_ok=True)

    # Zapisz obrazy
    for img_file in images:
        img_path = os.path.join(image_folder, img_file.filename)
        with open(img_path, "wb") as f:
            f.write(await img_file.read())
        logger.debug(f"[DEBUG] Zapisano obraz: {img_path}")

    # Zapisz annotacje
    for ann_file in annotations:
        ann_path = os.path.join(annotation_path, ann_file.filename)
        with open(ann_path, "wb") as f:
            f.write(await ann_file.read())
        logger.debug(f"[DEBUG] Zapisano annotację: {ann_path}")

    return {"message": "Dane przygotowane pomyślnie"}

@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest, http_request: Request):
    user_role = http_request.headers.get("X-User-Role")
    logger.debug(f"[DEBUG] Wywołanie /benchmark: user_role={user_role}, algorithm={request.algorithm}, model_version={request.model_version}, model_name={request.model_name}")
    if user_role != "admin":
        logger.error("[DEBUG] Brak uprawnień: użytkownik nie jest adminem")
        raise HTTPException(status_code=403, detail="Only admins can run benchmark")

    # Sprawdź, czy podane ścieżki istnieją
    if not os.path.exists(request.image_folder):
        logger.error(f"[DEBUG] Folder z obrazami nie istnieje: {request.image_folder}")
        raise HTTPException(status_code=400, detail=f"Folder z obrazami nie istnieje: {request.image_folder}")

    if not os.path.exists(request.annotation_path):
        logger.error(f"[DEBUG] Folder z annotacjami nie istnieje: {request.annotation_path}")
        raise HTTPException(status_code=400, detail=f"Folder z annotacjami nie istnieje: {request.annotation_path}")

    # Wczytaj dane za pomocą ImageDataset
    dataset = ImageDataset(request.image_folder, request.annotation_path)
    if len(dataset) == 0:
        logger.error("[DEBUG] Brak obrazów w podanym folderze")
        raise HTTPException(status_code=400, detail="Brak obrazów w podanym folderze")

    # Mapowanie katalogów dla każdego algorytmu
    algorithm_to_path = {
        "Mask R-CNN": "/app/backend/Mask_RCNN/data/test/images",
        "MCNN": "/app/backend/MCNN/data/test/images",
        "FasterRCNN": "/app/backend/FasterRCNN/data/test/images",
    }
    if request.algorithm not in algorithm_to_path:
        logger.error(f"[DEBUG] Algorytm {request.algorithm} nie jest obsługiwany")
        raise HTTPException(status_code=400, detail=f"Algorytm {request.algorithm} nie jest obsługiwany")

    # Sprawdź, czy model istnieje
    model_paths = {
        "Mask R-CNN": "/app/backend/Mask_RCNN/models/",
        "MCNN": "/app/backend/MCNN/models/",
        "FasterRCNN": "/app/backend/FasterRCNN/models/",
    }
    # Używamy model_name do skonstruowania nazwy pliku modelu
    file_name = f"{request.model_name}_checkpoint.pth"
    model_path = os.path.join(model_paths.get(request.algorithm, ""), file_name)
    if not os.path.exists(model_path):
        logger.error(f"[DEBUG] Plik modelu nie istnieje: {model_path}")
        raise HTTPException(status_code=400, detail=f"Plik modelu nie istnieje: {model_path}")

    # Tworzenie katalogu na obrazy, jeśli nie istnieje
    container_images_path = algorithm_to_path[request.algorithm]
    os.makedirs(container_images_path, exist_ok=True)

    metrics_list = []
    for idx in range(len(dataset)):
        img_path, annotations = dataset[idx]
        logger.debug(f"[DEBUG] Przetwarzanie obrazu {idx+1}/{len(dataset)}: {img_path}")

        # Kopiowanie obrazu do katalogu algorytmu
        container_image_path = os.path.join(container_images_path, os.path.basename(img_path))
        logger.debug(f"[DEBUG] Kopiowanie obrazu z {img_path} do {container_image_path}")
        shutil.copy(img_path, container_image_path)

        try:
            # Uruchom detekcję przy użyciu metody analyze_with_model
            logger.debug(f"[DEBUG] Uruchamianie detekcji na obrazie {os.path.basename(img_path)}")
            result, num_predicted = detection_api.analyze_with_model(container_image_path, request.algorithm, file_name)  # Używamy skonstruowanego file_name
            if "Błąd" in result:
                logger.error(f"[DEBUG] Błąd detekcji dla {img_path}: {result}")
                raise HTTPException(status_code=500, detail=f"Błąd detekcji dla {img_path}: {result}")
            logger.debug(f"[DEBUG] Liczba wykrytych rur: {num_predicted} dla {img_path}")

            # Policz obiekty z annotacji (format LabelMe)
            num_ground_truth = len(annotations)
            logger.debug(f"[DEBUG] Liczba rur w annotacji: {num_ground_truth} dla {img_path}")

            # Oblicz metrykę
            metric = abs(num_predicted - num_ground_truth)
            metrics_list.append(metric)
            logger.debug(f"[DEBUG] Metryka dla {img_path}: {metric}")
        except Exception as e:
            logger.error(f"[DEBUG] Błąd podczas przetwarzania {img_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania {img_path}: {e}")
        finally:
            # Usuń skopiowany obraz
            if os.path.exists(container_image_path):
                os.unlink(container_image_path)

    if metrics_list:
        mae = sum(metrics_list) / len(metrics_list)
        results = {"MAE": mae, "algorithm": request.algorithm, "model_version": request.model_version}
        logger.debug(f"[DEBUG] Wynik benchmarku: MAE={mae}")
        try:
            with open("/app/backend/benchmark_results.json", "w") as f:
                json.dump(results, f)
            logger.debug("[DEBUG] Wyniki zapisane do pliku")
        except Exception as e:
            logger.error(f"[DEBUG] Błąd zapisu wyników: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd zapisu wyników: {e}")
        return results
    else:
        logger.error("[DEBUG] Brak przetworzonych par obraz-annotacja")
        raise HTTPException(status_code=400, detail="No valid image-annotation pairs processed")

@app.get("/get_benchmark_results")
async def get_benchmark_results(request: Request):
    user_role = request.headers.get("X-User-Role")
    logger.debug(f"[DEBUG] Wywołanie /get_benchmark_results: user_role={user_role}")
    if user_role is None:
        logger.error("[DEBUG] Brak nagłówka X-User-Role")
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        with open("/app/backend/benchmark_results.json", "r") as f:
            results = json.load(f)
        logger.debug(f"[DEBUG] Zwrócono wyniki: {results}")
        return results
    except FileNotFoundError:
        logger.error("[DEBUG] Brak pliku z wynikami benchmarku")
        raise HTTPException(status_code=404, detail="No benchmark results available")
    except Exception as e:
        logger.error(f"[DEBUG] Błąd podczas odczytu wyników: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd podczas odczytu wyników: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)