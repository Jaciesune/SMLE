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
from datetime import datetime

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

@app.get("/compare_models")
def compare_models():
    history_file = "/app/backend/benchmark_history.json"
    if not os.path.exists(history_file):
        logger.debug("[DEBUG] Brak historii benchmarków")
        return {"results": [], "best_model": None}

    try:
        with open(history_file, "r") as f:
            history = json.load(f)
    except Exception as e:
        logger.error(f"[DEBUG] Błąd odczytu historii benchmarków: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd odczytu historii: {e}")

    if not history:
        logger.debug("[DEBUG] Historia benchmarków jest pusta")
        return {"results": [], "best_model": None}

    # Grupowanie wyników według folderu danych
    results_by_dataset = {}
    for result in history:
        dataset = result.get("image_folder", "unknown_dataset")
        if dataset not in results_by_dataset:
            results_by_dataset[dataset] = []
        results_by_dataset[dataset].append(result)

    # Przygotowanie wyników porównania
    comparison_results = []
    best_model_info = None
    overall_best_effectiveness = -1

    for dataset, results in results_by_dataset.items():
        dataset_results = []
        best_effectiveness = -1
        best_model_for_dataset = None

        for result in results:
            model_info = {
                "model": f"{result.get('algorithm', 'Unknown')} - v{result.get('model_version', 'Unknown')}",
                "model_name": result.get("model_name", "Unknown"),
                "effectiveness": result.get("effectiveness", 0),
                "mae": result.get("MAE", 0),
                "timestamp": result.get("timestamp", "Unknown")
            }
            dataset_results.append(model_info)

            # Znajdź najlepszy model dla tego zbioru danych
            effectiveness = result.get("effectiveness", 0)
            if effectiveness > best_effectiveness:
                best_effectiveness = effectiveness
                best_model_for_dataset = {
                    "dataset": dataset,
                    "model": f"{result.get('algorithm', 'Unknown')} - v{result.get('model_version', 'Unknown')}",
                    "model_name": result.get("model_name", "Unknown"),
                    "effectiveness": effectiveness,
                    "mae": result.get("MAE", 0)
                }

        # Sortuj wyniki według skuteczności (od najlepszego do najgorszego)
        dataset_results.sort(key=lambda x: x["effectiveness"], reverse=True)

        # Dodaj ranking i różnice skuteczności
        for i, model in enumerate(dataset_results):
            model["rank"] = i + 1
            if best_model_for_dataset:
                model["effectiveness_diff"] = best_effectiveness - model["effectiveness"]

        # Dodaj wyniki dla tego zbioru danych
        comparison_results.append({
            "dataset": dataset,
            "results": dataset_results,
            "best_model": best_model_for_dataset
        })

        # Aktualizuj najlepszy model ogólny
        if best_effectiveness > overall_best_effectiveness:
            overall_best_effectiveness = best_effectiveness
            best_model_info = best_model_for_dataset

    return {
        "results": comparison_results,
        "best_model": best_model_info
    }
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
    file_name = f"{request.model_name}_checkpoint.pth"
    model_path = os.path.join(model_paths.get(request.algorithm, ""), file_name)
    if not os.path.exists(model_path):
        logger.error(f"[DEBUG] Plik modelu nie istnieje: {model_path}")
        raise HTTPException(status_code=400, detail=f"Plik modelu nie istnieje: {model_path}")

    # Tworzenie katalogu na obrazy, jeśli nie istnieje
    container_images_path = algorithm_to_path[request.algorithm]
    os.makedirs(container_images_path, exist_ok=True)

    metrics_list = []
    ground_truth_counts = []  # Lista przechowująca liczby obiektów w ground truth
    predicted_counts = []  # Lista przechowująca liczby przewidywanych obiektów
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
            result, num_predicted = detection_api.analyze_with_model(container_image_path, request.algorithm, file_name)
            if "Błąd" in result:
                logger.error(f"[DEBUG] Błąd detekcji dla {img_path}: {result}")
                raise HTTPException(status_code=500, detail=f"Błąd detekcji dla {img_path}: {result}")
            logger.debug(f"[DEBUG] Liczba wykrytych rur: {num_predicted} dla {img_path}")

            # Policz obiekty z annotacji (format LabelMe)
            num_ground_truth = len(annotations)
            ground_truth_counts.append(num_ground_truth)
            predicted_counts.append(num_predicted)
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
        # Oblicz MAE
        mae = sum(metrics_list) / len(metrics_list)
        # Oblicz średnią liczbę obiektów w ground truth
        avg_ground_truth = sum(ground_truth_counts) / len(ground_truth_counts) if ground_truth_counts else 1
        logger.debug(f"[DEBUG] Średnia liczba obiektów w ground truth: {avg_ground_truth}")
        # Oblicz średnią liczbę przewidywanych obiektów
        avg_predicted = sum(predicted_counts) / len(predicted_counts) if predicted_counts else 0
        logger.debug(f"[DEBUG] Średnia liczba przewidywanych obiektów: {avg_predicted}")
        # Oblicz skuteczność w procentach
        effectiveness = max(0, (1 - mae / avg_ground_truth)) * 100 if avg_ground_truth > 0 else 0
        results = {
            "MAE": mae,
            "effectiveness": round(effectiveness, 2),  # Skuteczność w procentach, zaokrąglona do 2 miejsc
            "algorithm": request.algorithm,
            "model_version": request.model_version,
            "model_name": request.model_name,
            "image_folder": request.image_folder,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"[DEBUG] Wynik benchmarku: MAE={mae}, Skuteczność={effectiveness}%, avg_ground_truth={avg_ground_truth}, avg_predicted={avg_predicted}")

        # Aktualizuj accuracy w bazie danych
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = """
                UPDATE model 
                SET accuracy = %s 
                WHERE name = %s AND algorithm = %s AND version = %s
            """
            cursor.execute(query, (effectiveness, request.model_name, request.algorithm, request.model_version))
            conn.commit()
            logger.debug(f"[DEBUG] Zaktualizowano accuracy w bazie danych: {effectiveness}% dla modelu {request.model_name}")
        except mysql.connector.Error as err:
            logger.error(f"[DEBUG] Błąd aktualizacji accuracy w bazie danych: {err}")
            raise HTTPException(status_code=500, detail=f"Błąd aktualizacji accuracy: {err}")
        finally:
            cursor.close()
            conn.close()

        # Zapisz wyniki do pliku benchmark_results.json (nadpisywanie pojedynczego wyniku)
        try:
            with open("/app/backend/benchmark_results.json", "w") as f:
                json.dump(results, f)
            logger.debug("[DEBUG] Wyniki zapisane do pliku benchmark_results.json")
        except Exception as e:
            logger.error(f"[DEBUG] Błąd zapisu wyników: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd zapisu wyników: {e}")

        # Zapisz wyniki do historii benchmarków (nadpisywanie dla tych samych danych)
        try:
            history_file = "/app/backend/benchmark_history.json"
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r") as f:
                        history = json.load(f)
                        if not isinstance(history, list):
                            history = []
                except json.JSONDecodeError:
                    history = []  # Reset w przypadku błędnego formatu JSON

            # Szukaj istniejącego wpisu dla tych samych danych
            found = False
            for i, entry in enumerate(history):
                if (entry.get("model_name") == request.model_name and
                    entry.get("algorithm") == request.algorithm and
                    entry.get("model_version") == request.model_version and
                    entry.get("image_folder") == request.image_folder):
                    history[i] = results  # Nadpisz istniejący wpis
                    found = True
                    logger.debug(f"[DEBUG] Nadpisz istniejący wpis w historii dla modelu {request.model_name}")
                    break

            if not found:
                history.append(results)  # Dodaj nowy wpis, jeśli nie znaleziono dopasowania
                logger.debug(f"[DEBUG] Dodano nowy wpis do historii dla modelu {request.model_name}")

            with open(history_file, "w") as f:
                json.dump(history, f, indent=4)
            logger.debug("[DEBUG] Zaktualizowano historię benchmarków")
        except Exception as e:
            logger.error(f"[DEBUG] Błąd zapisu historii benchmarków: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd zapisu historii: {e}")
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
        history_file = "/app/backend/benchmark_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        else:
            history = []
        logger.debug(f"[DEBUG] Zwrócono historię: {history}")
        return {"history": history}
    except FileNotFoundError:
        logger.error("[DEBUG] Brak pliku z wynikami benchmarku")
        raise HTTPException(status_code=404, detail="No benchmark results available")
    except Exception as e:
        logger.error(f"[DEBUG] Błąd podczas odczytu wyników: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd podczas odczytu wyników: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)