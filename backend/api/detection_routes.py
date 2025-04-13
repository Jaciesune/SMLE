import os
import shutil
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from api.detection_api import DetectionAPI

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicjalizacja routera
router = APIRouter()

# Inicjalizacja DetectionAPI
detection_api = DetectionAPI()

@router.get("/detect_algorithms")
def get_algorithms():
    """Zwraca listę dostępnych algorytmów."""
    logger.debug("Otrzymano żądanie GET /detect_algorithms")
    algorithms = detection_api.get_algorithms()
    logger.debug("Zwracam algorytmy: %s", algorithms)
    return algorithms

@router.get("/detect_model_versions/{algorithm}")
def get_model_versions(algorithm: str):
    """Zwraca listę wersji modeli dla wybranego algorytmu."""
    logger.debug("Otrzymano żądanie GET /detect_model_versions/%s", algorithm)
    versions = detection_api.get_model_versions(algorithm)
    logger.debug("Zwracam wersje modeli dla %s: %s", algorithm, versions)
    return versions
    
@router.post("/detect_image")
async def detect_image(
    algorithm: str = Form(...),
    model_version: str = Form(...),
    image: UploadFile = File(...)
):
    logger.debug("Rozpoczynam detekcję: algorithm=%s, model_version=%s, image=%s",
                 algorithm, model_version, image.filename)

    # Dynamiczne ścieżki zależne od algorytmu
    if algorithm == "Mask R-CNN":
        input_dir_docker = "/app/backend/Mask_RCNN/data/test/images"
        output_dir_docker = "/app/backend/Mask_RCNN/data/detectes"
    elif algorithm == "FasterRCNN":
        input_dir_docker = "/app/backend/FasterRCNN/data/test/images"
        output_dir_docker = "/app/backend/FasterRCNN/data/detectes"
    elif algorithm == "MCNN":
        input_dir_docker = "/app/backend/MCNN/data/test/images"
        output_dir_docker = "/app/backend/MCNN/data/detectes"
    else:
        logger.error("Nieobsługiwany algorytm: %s", algorithm)
        raise HTTPException(status_code=400, detail=f"Nieobsługiwany algorytm: {algorithm}")

    image_path_docker = None
    try:
        # Tworzenie katalogów
        os.makedirs(input_dir_docker, exist_ok=True)
        os.makedirs(output_dir_docker, exist_ok=True)

        # Zapisz obraz z oryginalną nazwą
        image_path_docker = os.path.join(input_dir_docker, image.filename)
        with open(image_path_docker, "wb") as f:
            shutil.copyfileobj(image.file, f)
        logger.debug("Zapisano zdjęcie: %s", image_path_docker)

        # Weryfikacja, czy plik istnieje po zapisie
        if not os.path.exists(image_path_docker):
            logger.error("Plik %s nie został zapisany!", image_path_docker)
            raise HTTPException(status_code=500, detail=f"Nie udało się zapisać obrazu: {image_path_docker}")
        logger.debug("Plik %s istnieje po zapisie.", image_path_docker)

        # Wykonaj detekcję za pomocą DetectionAPI
        result, detections_count = detection_api.analyze_with_model(
            image_path_docker, algorithm, model_version
        )
        logger.debug("Wynik detekcji: result=%s, detections_count=%d", result, detections_count)

        if "Błąd" in result:
            logger.error("Błąd w detekcji: %s", result)
            raise HTTPException(status_code=500, detail=result)

        # Sprawdzenie, czy plik wynikowy istnieje
        if not os.path.exists(result):
            logger.error("Plik wynikowy %s nie istnieje.", result)
            raise HTTPException(status_code=500, detail=f"Plik wynikowy {result} nie został utworzony.")

        # Zwracanie obrazu z liczbą detekcji w nagłówku
        response = FileResponse(result, filename=os.path.basename(result))
        response.headers["X-Detections-Count"] = str(detections_count)
        return response
    except Exception as e:
        logger.error("Błąd w endpointcie /detect_image: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas detekcji: {str(e)}")
    finally:
        # Zamknij plik
        await image.close()
        # Usunięcie tymczasowego pliku wejściowego tylko w przypadku sukcesu
        # (przeniesione do bloku po detekcji, aby nie usuwać pliku w razie błędu)