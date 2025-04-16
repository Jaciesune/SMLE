import os
import shutil
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from api.detection_api import DetectionAPI
from archive_tab import get_db_connection

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("api.detection_routes")

router = APIRouter()
detection_api = DetectionAPI()

@router.get("/detect_algorithms")
def get_algorithms():
    """Zwraca listę dostępnych algorytmów."""
    algs = detection_api.get_algorithms()
    logger.debug("Zwracam algorytmy: %s", algs)
    return algs

@router.get("/detect_model_versions/{algorithm}")
def get_model_versions(algorithm: str):
    """Zwraca listę wersji modeli dla danego algorytmu."""
    versions = detection_api.get_model_versions(algorithm)
    logger.debug("Zwracam wersje modeli dla %s: %s", algorithm, versions)
    return versions

@router.post("/detect_image")
async def detect_image(
    algorithm: str = Form(...),
    model_version: str = Form(...),
    image: UploadFile = File(...),
    username: str = Form(None)  # opcjonalnie: nazwa zalogowanego użytkownika
):
    logger.debug(
        "Rozpoczynam detekcję: algorithm=%s, model_version=%s, image=%s, username=%s",
        algorithm, model_version, image.filename, username
    )

    # wybór ścieżek
    if algorithm == "Mask R-CNN":
        input_dir = "/app/backend/Mask_RCNN/data/test/images"
        output_dir = "/app/backend/Mask_RCNN/data/detectes"
    elif algorithm == "FasterRCNN":
        input_dir = "/app/backend/FasterRCNN/data/test/images"
        output_dir = "/app/backend/FasterRCNN/data/detectes"
    elif algorithm == "MCNN":
        input_dir = "/app/backend/MCNN/data/test/images"
        output_dir = "/app/backend/MCNN/data/detectes"
    else:
        raise HTTPException(400, f"Nieobsługiwany algorytm: {algorithm}")

    try:
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        img_path = os.path.join(input_dir, image.filename)
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        logger.debug("Zapisano zdjęcie: %s", img_path)

        # uruchomienie detekcji
        result_path, count = detection_api.analyze_with_model(
            img_path, algorithm, model_version
        )
        logger.debug("Wynik detekcji: result=%s, count=%d", result_path, count)

        # sprawdzenie błędów
        if "Błąd" in result_path or not os.path.exists(result_path):
            raise HTTPException(500, f"Detekcja nie powiodła się albo wynik nie istnieje: {result_path}")

        # zapis do archiwum
        if username:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT id FROM user WHERE name = %s", (username,))
                row = cur.fetchone()
                if row:
                    user_id = row[0]
                else:
                    raise Exception(f"Nie znaleziono użytkownika: {username}")
                model_id = 1  # na sztywno
                cur.execute(
                    "INSERT INTO archive(action, user_id, model_id, date) VALUES (%s,%s,%s,NOW())",
                    ("Detekcja obrazu", user_id, model_id)
                )
                conn.commit()
                cur.close()
                conn.close()
                logger.debug("Użytkownik przesłany do detekcji: %s", username)
                logger.debug("Zapisano do archive: user_id=%s, model_id=%s", user_id, model_id)
            except Exception as e:
                logger.error("Nie udało się zapisać do archiwum: %s", e)
        else:
            logger.warning("Nie podano username, pomijam zapis do archiwum.")

        # odpowiedź
        resp = FileResponse(result_path, filename=os.path.basename(result_path))
        resp.headers["X-Detections-Count"] = str(count)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Błąd w endpointcie /detect_image")
        raise HTTPException(500, str(e))
    finally:
        await image.close()
