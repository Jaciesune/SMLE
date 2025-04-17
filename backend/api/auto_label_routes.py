import os
import shutil
import zipfile
import logging
import stat
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from api.auto_label_api import AutoLabelAPI
from glob import glob

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicjalizacja routera
router = APIRouter()

# Inicjalizacja AutoLabelAPI
auto_label_api = AutoLabelAPI()

@router.post("/auto_label")
async def auto_label(
    background_tasks: BackgroundTasks,
    job_name: str = Form(...),
    model_version: str = Form(...),
    images: list[UploadFile] = File(...)
):
    logger.debug("Rozpoczynam auto_label dla job_name=%s, model_version=%s, %d obrazów",
                 job_name, model_version, len(images))
    input_dir_docker = f"/app/backend/data/Auto_labeling/{job_name}_before"
    output_dir_docker = f"/app/backend/data/Auto_labeling/{job_name}_after"
    debug_dir_docker = f"/app/backend/data/Auto_labeling/{job_name}_debug"

    try:
        # Zapisz przesłane obrazy
        os.makedirs(input_dir_docker, exist_ok=True)
        for image in images:
            image_path = os.path.join(input_dir_docker, image.filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            logger.debug("Zapisano obraz: %s", image_path)

        logger.debug("Wywołuję auto_label_api.auto_label...")
        result = auto_label_api.auto_label(
            input_dir_docker, job_name, model_version,
            input_dir_docker, output_dir_docker, debug_dir_docker
        )
        logger.debug("Wynik auto_label: %s", result)
        if "Błąd" in result:
            logger.error("Błąd w auto_label: %s", result)
            raise HTTPException(status_code=500, detail=result)

        return {"status": "success", "message": "Labelowanie zakończone pomyślnie", "job_name": job_name}
    except Exception as e:
        logger.error("Błąd w endpointcie /auto_label: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas auto-labelingu: {str(e)}")
    finally:
        for image in images:
            await image.close()

@router.get("/get_results/{job_name}")
async def get_results(job_name: str):
    output_dir = f"/app/backend/data/Auto_labeling/{job_name}_after"
    debug_dir = f"/app/backend/data/Auto_labeling/{job_name}_debug"
    zip_path = f"/app/backend/data/Auto_labeling/{job_name}_results.zip"

    try:
        # Utwórz ZIP z folderów tymczasowych
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(output_dir):
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(f"{job_name}_after", os.path.relpath(file_path, output_dir))
                        zipf.write(file_path, arcname)
                        logger.debug("Dodano do zip: %s", arcname)
            if os.path.exists(debug_dir):
                for root, _, files in os.walk(debug_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(f"{job_name}_debug", os.path.relpath(file_path, debug_dir))
                        zipf.write(file_path, arcname)
                        logger.debug("Dodano do zip: %s", arcname)

        # Ustaw uprawnienia do pliku ZIP
        os.chmod(zip_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        logger.debug("Ustawiono uprawnienia dla %s", zip_path)

        return FileResponse(zip_path, filename=f"{job_name}_results.zip")
    except Exception as e:
        logger.error("Błąd podczas tworzenia wyników: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia wyników: {str(e)}")

@router.get("/auto_label_jobs")
def get_auto_label_jobs():
    zip_pattern = "/app/backend/data/Auto_labeling/*_results.zip"
    zip_files = glob(zip_pattern)
    job_names = [os.path.splitext(os.path.basename(f))[0].replace("_results", "") for f in zip_files]
    logger.debug("Znalezione zadania: %s", job_names)
    return job_names

@router.get("/model_versions_maskrcnn")
def get_model_versions():
    versions = auto_label_api.get_model_versions()
    logger.debug("Dostępne wersje modeli Mask R-CNN: %s", versions)
    return versions