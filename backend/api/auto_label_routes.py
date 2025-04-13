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
    zip_path = f"/app/backend/data/Auto_labeling/{job_name}_results.zip"

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

        # Tworzenie zip bezpośrednio z wyników
        logger.debug("Pakuję wyniki do %s...", zip_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(output_dir_docker):
                for root, _, files in os.walk(output_dir_docker):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(f"{job_name}_after", os.path.relpath(file_path, output_dir_docker))
                        zipf.write(file_path, arcname)
                        logger.debug("Dodano do zip: %s", arcname)
            else:
                logger.warning("Katalog wyjściowy %s nie istnieje.", output_dir_docker)

            if os.path.exists(debug_dir_docker):
                for root, _, files in os.walk(debug_dir_docker):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(f"{job_name}_debug", os.path.relpath(file_path, debug_dir_docker))
                        zipf.write(file_path, arcname)
                        logger.debug("Dodano do zip: %s", arcname)
            else:
                logger.warning("Katalog debug %s nie istnieje.", debug_dir_docker)

        # Ustaw uprawnienia do pliku zip (odczyt/zapis dla wszystkich)
        try:
            os.chmod(zip_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)
            logger.debug("Ustawiono uprawnienia dla %s", zip_path)
        except Exception as e:
            logger.warning("Nie udało się ustawić uprawnień dla %s: %s", zip_path, e)

        # Weryfikacja istnienia pliku zip
        if not os.path.exists(zip_path):
            logger.error("Plik %s nie został utworzony.", zip_path)
            raise HTTPException(status_code=500, detail=f"Plik {zip_path} nie został utworzony.")

        # Usuwanie folderów po spakowaniu
        logger.debug("Dodaję zadania czyszczenia w tle...")
        background_tasks.add_task(shutil.rmtree, input_dir_docker, ignore_errors=True)
        background_tasks.add_task(shutil.rmtree, output_dir_docker, ignore_errors=True)
        background_tasks.add_task(shutil.rmtree, debug_dir_docker, ignore_errors=True)

        logger.debug("Zwracam plik %s", zip_path)
        return FileResponse(zip_path, filename=f"{job_name}_results.zip")
    except Exception as e:
        logger.error("Błąd w endpointcie /auto_label: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas auto-labelingu: {str(e)}")
    finally:
        # Zamknij wszystkie przesłane pliki
        for image in images:
            await image.close()

@router.get("/download_results/{job_name}")
async def download_results(job_name: str):
    zip_path = f"/app/backend/data/Auto_labeling/{job_name}_results.zip"
    logger.debug("Próba dostępu do pliku %s", zip_path)
    if not os.path.exists(zip_path):
        logger.error("Plik %s nie istnieje.", zip_path)
        raise HTTPException(status_code=404, detail=f"Plik wyników dla zadania {job_name} nie istnieje.")
    try:
        # Weryfikacja uprawnień
        if not os.access(zip_path, os.R_OK):
            logger.error("Brak uprawnień do odczytu pliku %s", zip_path)
            raise HTTPException(status_code=403, detail=f"Brak uprawnień do pliku {zip_path}")
        logger.debug("Zwracam plik %s", zip_path)
        return FileResponse(zip_path, filename=f"{job_name}_results.zip")
    except Exception as e:
        logger.error("Błąd podczas zwracania pliku %s: %s", zip_path, str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")

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