import os
import zipfile
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from api.dataset_api import DatasetAPI

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicjalizacja routera
router = APIRouter()

# Inicjalizacja DatasetAPI
dataset_api = DatasetAPI()

@router.post("/create_dataset")
async def create_dataset(
    background_tasks: BackgroundTasks,
    job_name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    logger.debug("Rozpoczynam tworzenie datasetu dla job_name=%s, %d plików", job_name, len(files))
    # Logowanie nazw przesłanych plików
    logger.debug("Przesłane pliki: %s", [file.filename for file in files])
    zip_path = f"/app/backend/data/Dataset_creation/{job_name}_results.zip"

    try:
        # Wywołaj DatasetAPI
        result_zip = dataset_api.create_dataset(job_name, files)
        logger.debug("Wynik create_dataset: %s", result_zip)
        if not os.path.exists(result_zip):
            logger.error("Plik %s nie został utworzony.", result_zip)
            raise HTTPException(status_code=500, detail="Nie udało się utworzyć datasetu.")

        logger.debug("Zwracam plik %s", result_zip)
        return FileResponse(result_zip, filename=f"{job_name}_results.zip")
    except Exception as e:
        logger.error("Błąd w endpointcie /create_dataset: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia datasetu: {str(e)}")
    finally:
        # Zamknij wszystkie przesłane pliki
        for file in files:
            await file.close()

@router.get("/download_dataset/{job_name}")
async def download_dataset(job_name: str):
    zip_path = f"/app/backend/data/Dataset_creation/{job_name}_results.zip"
    logger.debug("Próba dostępu do pliku %s", zip_path)
    if not os.path.exists(zip_path):
        logger.error("Plik %s nie istnieje.", zip_path)
        raise HTTPException(status_code=404, detail=f"Plik wyników dla zadania {job_name} nie istnieje.")
    try:
        logger.debug("Zwracam plik %s", zip_path)
        return FileResponse(zip_path, filename=f"{job_name}_results.zip")
    except Exception as e:
        logger.error("Błąd podczas zwracania pliku %s: %s", zip_path, str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")