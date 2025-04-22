import os
import shutil
import zipfile
import logging
import stat
import re
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from api.auto_label_api import AutoLabelAPI
from glob import glob
import json

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicjalizacja routera
router = APIRouter()

# Inicjalizacja AutoLabelAPI
auto_label_api = AutoLabelAPI()

def sanitize_filename(filename: str) -> str:
    """Czyści nazwę pliku, usuwając znaki specjalne i zamieniając spacje na podkreślniki."""
    filename = filename.replace(" ", "_")
    filename = re.sub(r'[^a-zA-Z0-9_\.]', '', filename)
    return filename

@router.post("/auto_label")
async def auto_label(
    background_tasks: BackgroundTasks,
    job_name: str = Form(...),
    model_version: str = Form(...),
    custom_label: str = Form(...),  # Nowy parametr
    images: list[UploadFile] = File(...)
):
    logger.debug("Rozpoczynam auto_label dla job_name=%s, model_version=%s, custom_label=%s, %d obrazów",
                 job_name, model_version, custom_label, len(images))
    input_dir_docker = f"/app/backend/data/Auto_labeling/{job_name}_before"
    output_dir_docker = f"/app/backend/data/Auto_labeling/{job_name}_after"

    try:
        # Utwórz katalogi (bez debug)
        for directory in [input_dir_docker, output_dir_docker]:
            try:
                os.makedirs(directory, exist_ok=True)
                os.chmod(directory, 0o777)  # Ustaw uprawnienia
                logger.debug(f"Utworzono katalog: {directory}")
            except Exception as e:
                logger.error(f"Błąd podczas tworzenia katalogu {directory}: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia katalogu {directory}: {e}")

        # Zapisz przesłane obrazy
        image_paths = []
        for image in images:
            sanitized_filename = sanitize_filename(image.filename)
            if not sanitized_filename.lower().endswith('.jpg'):
                logger.error(f"Plik {sanitized_filename} nie jest w formacie .jpg")
                raise HTTPException(status_code=400, detail=f"Plik {image.filename} nie jest w formacie .jpg")

            image_path = os.path.join(input_dir_docker, sanitized_filename)
            try:
                with open(image_path, "wb") as f:
                    shutil.copyfileobj(image.file, f)
                os.chmod(image_path, 0o666)  # Ustaw uprawnienia dla pliku
                image_paths.append(image_path)
                logger.debug("Zapisano obraz: %s", image_path)
            except Exception as e:
                logger.error(f"Błąd podczas zapisywania obrazu {image_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd podczas zapisywania obrazu {sanitized_filename}: {e}")

        logger.debug("Wywołuję auto_label_api.auto_label...")
        result = auto_label_api.auto_label(
            input_dir_docker, job_name, model_version,
            input_dir_docker, output_dir_docker, debug_dir_docker=None, custom_label=custom_label
        )
        logger.debug("Wynik auto_label: %s", result)
        if "Błąd" in result:
            if "Brak wyników" in result:
                logger.warning(f"Auto-labeling nie wygenerował wyników: {result}")
                return {"status": "success", "job_name": job_name, "message": "Nie znaleziono obiektów do oznaczenia."}
            else:
                logger.error("Błąd w auto_label: %s", result)
                raise HTTPException(status_code=500, detail=result)

        # Przetwórz wyniki, aby usunąć polygon_points
        annotation_files = glob(os.path.join(output_dir_docker, "*.json"))
        logger.debug(f"Znalezione pliki adnotacji: {annotation_files}")
        for annotation_path in annotation_files:
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                logger.debug(f"Wczytano adnotacje z {annotation_path}")
                
                # Usuń polygon_points z każdego shape
                for shape in annotation_data.get("shapes", []):
                    if "polygon_points" in shape:
                        del shape["polygon_points"]

                # Zapisz zmodyfikowane adnotacje
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation_data, f, indent=2)
                logger.debug(f"Zaktualizowano adnotacje w {annotation_path}")
            except Exception as e:
                logger.error(f"Błąd podczas przetwarzania adnotacji {annotation_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania adnotacji {annotation_path}: {e}")

        return {"status": "success", "message": "Labelowanie zakończone pomyślnie", "job_name": job_name}
    except Exception as e:
        logger.error("Błąd w endpointcie /auto_label: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas auto-labelingu: {str(e)}")
    finally:
        # Zamknij pliki obrazów
        for image in images:
            await image.close()
        # Usuń katalog input_dir_docker (before) po zakończeniu labelingu
        if os.path.exists(input_dir_docker):
            try:
                shutil.rmtree(input_dir_docker)
                logger.debug(f"Usunięto katalog before: {input_dir_docker}")
            except Exception as e:
                logger.error(f"Błąd podczas usuwania katalogu {input_dir_docker}: {e}")

@router.get("/get_results/{job_name}")
async def get_results(job_name: str):
    output_dir = f"/app/backend/data/Auto_labeling/{job_name}_after"
    zip_path = f"/app/backend/data/Auto_labeling/{job_name}_results.zip"

    try:
        # Utwórz ZIP tylko z katalogu output_dir
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(output_dir):
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(f"{job_name}_after", os.path.relpath(file_path, output_dir))
                        zipf.write(file_path, arcname)
                        logger.debug("Dodano do zip: %s", arcname)
            else:
                logger.error(f"Katalog wyników {output_dir} nie istnieje!")
                raise HTTPException(status_code=404, detail=f"Katalog wyników {output_dir} nie istnieje!")

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