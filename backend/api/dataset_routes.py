import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from api.dataset_api import DatasetAPI

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja routera
router = APIRouter()

# Inicjalizacja DatasetAPI
dataset_api = DatasetAPI()

@router.post("/create_dataset")
async def create_dataset(
    username: str = Form(...),
    job_name: str = Form(...),
    train_ratio: str = Form(...),  # Form(...) z typem str
    val_ratio: str = Form(...),    # Form(...) z typem str
    test_ratio: str = Form(...),   # Form(...) z typem str
    files: list[UploadFile] = File(...)
):
    logger.debug("Otrzymane parametry: username=%s, job_name=%s, train_ratio=%s, val_ratio=%s, test_ratio=%s",
                 username, job_name, train_ratio, val_ratio, test_ratio)
    logger.info("Rozpoczynam tworzenie datasetu dla username=%s, job_name=%s, %d plików", username, job_name, len(files))
    try:
        # Konwersja stringów na floaty
        train_ratio_float = float(train_ratio)
        val_ratio_float = float(val_ratio)
        test_ratio_float = float(test_ratio)
        dataset_api.create_dataset(username, job_name, files, train_ratio_float, val_ratio_float, test_ratio_float)
        logger.info("Dataset %s utworzony pomyślnie dla użytkownika %s", job_name, username)
        return {"status": "success"}
    except ValueError as e:
        logger.error("Błąd konwersji proporcji na liczby: %s", str(e))
        raise HTTPException(status_code=422, detail=f"Błąd walidacji proporcji: {str(e)}")
    except Exception as e:
        logger.error("Błąd w endpointcie /create_dataset: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia datasetu: {str(e)}")

@router.get("/list_datasets/{username}")
async def list_datasets(username: str):
    try:
        datasets = dataset_api.list_datasets(username)
        logger.info("Pobrano listę datasetów dla użytkownika %s: %s", username, datasets)
        return {"datasets": datasets}
    except Exception as e:
        logger.error("Błąd w endpointcie /list_datasets: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas pobierania listy datasetów: {str(e)}")

@router.get("/dataset_info/{username}/{dataset_name}")
async def dataset_info(username: str, dataset_name: str):
    try:
        info = dataset_api.get_dataset_info(username, dataset_name)
        logger.info("Pobrano informacje o datasecie %s dla użytkownika %s", dataset_name, username)
        return info
    except Exception as e:
        logger.error("Błąd w endpointcie /dataset_info: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas pobierania informacji o datasecie: {str(e)}")

@router.get("/download_dataset/{username}/{dataset_name}/{subset}")
async def download_dataset_subset(username: str, dataset_name: str, subset: str):
    try:
        zip_path = dataset_api.download_dataset(username, dataset_name, subset)
        logger.info("Pobrano podzbiór %s datasetu %s dla użytkownika %s", subset, dataset_name, username)
        return FileResponse(zip_path, filename=f"{dataset_name}_{subset}_results.zip")
    except Exception as e:
        logger.error("Błąd w endpointcie /download_dataset (subset): %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")

@router.get("/download_dataset/{username}/{dataset_name}")
async def download_dataset_full(username: str, dataset_name: str):
    try:
        zip_path = dataset_api.download_dataset(username, dataset_name)
        logger.info("Pobrano cały dataset %s dla użytkownika %s", dataset_name, username)
        return FileResponse(zip_path, filename=f"{dataset_name}_full_results.zip")
    except Exception as e:
        logger.error("Błąd w endpointcie /download_dataset (full): %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")

@router.delete("/delete_zip/{username}/{dataset_name}/{subset}")
async def delete_zip(username: str, dataset_name: str, subset: str):
    try:
        zip_path = os.path.join("/app/backend/data/dataset_create", username, dataset_name, f"{subset}_results.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("Usunięto plik ZIP: %s", zip_path)
        return {"message": "ZIP usunięty"}
    except Exception as e:
        logger.error("Błąd podczas usuwania ZIP: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania ZIP: {str(e)}")

@router.delete("/delete_zip/{username}/{dataset_name}")
async def delete_full_zip(username: str, dataset_name: str):
    try:
        zip_path = os.path.join("/app/backend/data/dataset_create", username, dataset_name, "full_results.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("Usunięto plik ZIP: %s", zip_path)
        return {"message": "ZIP usunięty"}
    except Exception as e:
        logger.error("Błąd podczas usuwania ZIP: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania ZIP: {str(e)}")

@router.delete("/delete_dataset/{username}/{dataset_name}")
async def delete_dataset(username: str, dataset_name: str):
    try:
        dataset_api.delete_dataset(username, dataset_name)
        logger.info("Usunięto dataset %s dla użytkownika %s", dataset_name, username)
        return {"status": "success"}
    except Exception as e:
        logger.error("Błąd w endpointcie /delete_dataset: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania datasetu: {str(e)}")