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
    username: str = Form(...),
    job_name: str = Form(...),
    train_ratio: float = Form(...),
    val_ratio: float = Form(...),
    test_ratio: float = Form(...),
    files: list[UploadFile] = File(...)
):
    logger.debug("Rozpoczynam tworzenie datasetu dla username=%s, job_name=%s, %d plików", username, job_name, len(files))
    logger.debug("Przesłane pliki: %s", [file.filename for file in files])
    zip_path = os.path.join("/app/backend/data/dataset_create", username, job_name, "results.zip")

    try:
        result_zip = dataset_api.create_dataset(username, job_name, files, train_ratio, val_ratio, test_ratio)
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
        for file in files:
            await file.close()

@router.get("/list_datasets/{username}")
async def list_datasets(username: str):
    try:
        datasets = dataset_api.list_datasets(username)
        return {"datasets": datasets}
    except Exception as e:
        logger.error("Błąd w endpointcie /list_datasets: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas pobierania listy datasetów: {str(e)}")

@router.get("/dataset_info/{username}/{dataset_name}")
async def dataset_info(username: str, dataset_name: str):
    try:
        info = dataset_api.get_dataset_info(username, dataset_name)
        return info
    except Exception as e:
        logger.error("Błąd w endpointcie /dataset_info: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas pobierania informacji o datasecie: {str(e)}")

@router.get("/download_dataset/{username}/{dataset_name}/{subset}")
async def download_dataset_subset(username: str, dataset_name: str, subset: str):
    try:
        zip_path = dataset_api.download_dataset(username, dataset_name, subset)
        return FileResponse(zip_path, filename=f"{dataset_name}_{subset}_results.zip")
    except Exception as e:
        logger.error("Błąd w endpointcie /download_dataset (subset): %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")

@router.get("/download_dataset/{username}/{dataset_name}")
async def download_dataset_full(username: str, dataset_name: str):
    try:
        zip_path = dataset_api.download_dataset(username, dataset_name)
        return FileResponse(zip_path, filename=f"{dataset_name}_results.zip")
    except Exception as e:
        logger.error("Błąd w endpointcie /download_dataset (full): %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")

@router.delete("/delete_zip/{username}/{dataset_name}/{subset}")
async def delete_zip(username: str, dataset_name: str, subset: str):
    try:
        zip_path = os.path.join("/app/backend/data/dataset_create", username, dataset_name, f"{subset}_results.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.debug("Usunięto %s", zip_path)
        return {"message": "ZIP usunięty"}
    except Exception as e:
        logger.error("Błąd podczas usuwania ZIP: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania ZIP: {str(e)}")

@router.delete("/delete_zip/{username}/{dataset_name}")
async def delete_full_zip(username: str, dataset_name: str):
    try:
        zip_path = os.path.join("/app/backend/data/dataset_create", username, dataset_name, "results.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.debug("Usunięto %s", zip_path)
        return {"message": "ZIP usunięty"}
    except Exception as e:
        logger.error("Błąd podczas usuwania ZIP: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania ZIP: {str(e)}")