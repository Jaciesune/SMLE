"""
Moduł tras API dla automatycznego oznaczania obrazów (Auto Label Routes)

Ten moduł dostarcza endpointy FastAPI do automatycznego oznaczania (etykietowania) obrazów
przy użyciu modeli Mask R-CNN. Umożliwia przesyłanie obrazów, ich automatyczne oznaczanie,
a następnie pobieranie wyników w postaci plików adnotacji w formacie LabelMe.
"""

#######################
# Importy bibliotek
#######################
import os                    # Do operacji na systemie plików
import shutil                # Do kopiowania i usuwania plików
import zipfile               # Do tworzenia archiwów ZIP
import logging               # Do logowania informacji i błędów
import stat                  # Do zarządzania uprawnieniami plików
import re                    # Do operacji wyrażeń regularnych
import json                  # Do operacji na plikach JSON
from glob import glob        # Do wyszukiwania plików według wzorca

from fastapi import (
    APIRouter,               # Klasa do tworzenia routerów w FastAPI
    HTTPException,           # Klasa do zgłaszania wyjątków HTTP
    BackgroundTasks,         # Klasa do zadań w tle
    UploadFile,              # Klasa do obsługi przesyłanych plików
    File,                    # Funkcja do deklarowania parametrów plików
    Form                     # Funkcja do deklarowania parametrów formularza
)
from fastapi.responses import FileResponse  # Klasa do zwracania plików jako odpowiedzi HTTP
from api.auto_label_api import AutoLabelAPI  # Import naszego API auto-labelingu

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#######################
# Inicjalizacja routera i API
#######################
router = APIRouter()
auto_label_api = AutoLabelAPI()

#######################
# Funkcje pomocnicze
#######################
def sanitize_filename(filename: str) -> str:
    """
    Czyści nazwę pliku z niedozwolonych znaków i spacji.
    
    Parameters:
        filename (str): Nazwa pliku do oczyszczenia
        
    Returns:
        str: Oczyszczona nazwa pliku zawierająca tylko litery, cyfry, podkreślenia i kropki
    """
    filename = filename.replace(" ", "_")  # Zamiana spacji na podkreślenia
    filename = re.sub(r'[^a-zA-Z0-9_\.]', '', filename)  # Usunięcie niepożądanych znaków
    return filename

#######################
# Endpointy API
#######################

@router.post("/auto_label")
async def auto_label(
    background_tasks: BackgroundTasks,
    job_name: str = Form(...),
    model_version: str = Form(...),
    custom_label: str = Form(...),
    images: list[UploadFile] = File(...)
):
    """
    Endpoint do automatycznego oznaczania obrazów.
    
    Przyjmuje przesłane obrazy, a następnie przeprowadza ich automatyczne oznaczanie
    przy użyciu określonego modelu Mask R-CNN i zwraca status operacji.
    
    Parameters:
        background_tasks (BackgroundTasks): Obiekt do wykonywania zadań w tle
        job_name (str): Nazwa zadania (do identyfikacji i późniejszego pobrania wyników)
        model_version (str): Wersja modelu Mask R-CNN do użycia
        custom_label (str): Niestandardowa etykieta do przypisania wykrytym obiektom
        images (list[UploadFile]): Lista przesłanych plików obrazów
        
    Returns:
        dict: Słownik zawierający informacje o statusie operacji:
            - status: "success" jeśli operacja się powiodła
            - message: Komunikat o rezultacie operacji
            - job_name: Nazwa zadania (do późniejszego pobrania wyników)
            
    Raises:
        HTTPException: 
            - 400: Gdy przesłany plik nie jest w formacie .jpg
            - 500: Gdy wystąpi błąd podczas przetwarzania plików lub oznaczania
    """
    #######################
    # Logowanie parametrów wejściowych
    #######################
    logger.debug("Rozpoczynam auto_label dla job_name=%s, model_version=%s, custom_label=%s, %d obrazów",
                 job_name, model_version, custom_label, len(images))
    
    #######################
    # Konfiguracja ścieżek
    #######################
    input_dir_docker = f"/app/backend/data/Auto_labeling/{job_name}_before"
    output_dir_docker = f"/app/backend/data/Auto_labeling/{job_name}_after"

    try:
        #######################
        # Tworzenie katalogów roboczych
        #######################
        for directory in [input_dir_docker, output_dir_docker]:
            try:
                os.makedirs(directory, exist_ok=True)
                os.chmod(directory, 0o777)  # Ustawienie uprawnień do odczytu, zapisu i wykonywania dla wszystkich
                logger.debug(f"Utworzono katalog: {directory}")
            except Exception as e:
                logger.error(f"Błąd podczas tworzenia katalogu {directory}: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia katalogu {directory}: {e}")

        #######################
        # Zapisywanie przesłanych obrazów
        #######################
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
                os.chmod(image_path, 0o666)  # Ustawienie uprawnień do odczytu i zapisu dla wszystkich
                image_paths.append(image_path)
                logger.debug("Zapisano obraz: %s", image_path)
            except Exception as e:
                logger.error(f"Błąd podczas zapisywania obrazu {image_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd podczas zapisywania obrazu {sanitized_filename}: {e}")

        #######################
        # Automatyczne oznaczanie obrazów
        #######################
        logger.debug("Wywołuję auto_label_api.auto_label...")
        result = auto_label_api.auto_label(
            input_dir_docker, job_name, model_version,
            input_dir_docker, output_dir_docker, debug_dir_docker=None, custom_label=custom_label
        )
        logger.debug("Wynik auto_label: %s", result)
        
        # Obsługa błędów i braku wyników
        if "Błąd" in result:
            if "Brak wyników" in result:
                logger.warning(f"Auto-labeling nie wygenerował wyników: {result}")
                return {"status": "success", "job_name": job_name, "message": "Nie znaleziono obiektów do oznaczenia."}
            else:
                logger.error("Błąd w auto_label: %s", result)
                raise HTTPException(status_code=500, detail=result)

        #######################
        # Przetwarzanie wygenerowanych adnotacji
        #######################
        annotation_files = glob(os.path.join(output_dir_docker, "*.json"))
        logger.debug(f"Znalezione pliki adnotacji: {annotation_files}")
        
        # Usuwanie niepotrzebnych pól z adnotacji
        for annotation_path in annotation_files:
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                logger.debug(f"Wczytano adnotacje z {annotation_path}")
                
                # Usunięcie pola polygon_points, które nie jest wymagane w formacie LabelMe
                for shape in annotation_data.get("shapes", []):
                    if "polygon_points" in shape:
                        del shape["polygon_points"]

                # Zapisanie zmodyfikowanych adnotacji
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation_data, f, indent=2)
                logger.debug(f"Zaktualizowano adnotacje w {annotation_path}")
            except Exception as e:
                logger.error(f"Błąd podczas przetwarzania adnotacji {annotation_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania adnotacji {annotation_path}: {e}")

        # Zwróć informację o sukcesie
        return {"status": "success", "message": "Labelowanie zakończone pomyślnie", "job_name": job_name}
    except Exception as e:
        # Obsługa ogólnych błędów
        logger.error("Błąd w endpointcie /auto_label: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas auto-labelingu: {str(e)}")
    finally:
        # Zamknij uchwyty plików obrazów
        for image in images:
            await image.close()
        
        # Usuń katalog wejściowy (by nie zajmować miejsca)
        if os.path.exists(input_dir_docker):
            try:
                shutil.rmtree(input_dir_docker)
                logger.debug(f"Usunięto katalog before: {input_dir_docker}")
            except Exception as e:
                logger.error(f"Błąd podczas usuwania katalogu {input_dir_docker}: {e}")

@router.get("/get_results/{job_name}")
async def get_results(job_name: str):
    """
    Endpoint do pobierania wyników automatycznego oznaczania obrazów.
    
    Tworzy plik ZIP z wynikami oznaczania dla określonego zadania
    i zwraca go jako odpowiedź HTTP.
    
    Parameters:
        job_name (str): Nazwa zadania, którego wyniki mają zostać pobrane
        
    Returns:
        FileResponse: Odpowiedź HTTP zawierająca plik ZIP z wynikami
        
    Raises:
        HTTPException: 
            - 404: Gdy katalog z wynikami nie istnieje
            - 500: Gdy wystąpi błąd podczas tworzenia pliku ZIP
    """
    #######################
    # Konfiguracja ścieżek
    #######################
    output_dir = f"/app/backend/data/Auto_labeling/{job_name}_after"
    zip_path = f"/app/backend/data/Auto_labeling/{job_name}_results.zip"

    try:
        #######################
        # Tworzenie archiwum ZIP z wynikami
        #######################
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Sprawdzenie, czy katalog z wynikami istnieje
            if os.path.exists(output_dir):
                # Dodanie wszystkich plików z katalogu wynikowego do archiwum
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(f"{job_name}_after", os.path.relpath(file_path, output_dir))
                        zipf.write(file_path, arcname)
                        logger.debug("Dodano do zip: %s", arcname)
            else:
                logger.error(f"Katalog wyników {output_dir} nie istnieje!")
                raise HTTPException(status_code=404, detail=f"Katalog wyników {output_dir} nie istnieje!")

        # Ustawienie odpowiednich uprawnień do pliku ZIP
        os.chmod(zip_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        logger.debug("Ustawiono uprawnienia dla %s", zip_path)

        # Zwróć plik ZIP jako odpowiedź HTTP
        return FileResponse(zip_path, filename=f"{job_name}_results.zip")
    except Exception as e:
        # Obsługa ogólnych błędów
        logger.error("Błąd podczas tworzenia wyników: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia wyników: {str(e)}")

@router.get("/auto_label_jobs")
def get_auto_label_jobs():
    """
    Endpoint do pobierania listy wykonanych zadań automatycznego oznaczania.
    
    Zwraca listę nazw zadań, dla których istnieją pliki wyników.
    
    Returns:
        list: Lista nazw zadań oznaczania
    """
    #######################
    # Wyszukiwanie plików wynikowych
    #######################
    zip_pattern = "/app/backend/data/Auto_labeling/*_results.zip"
    zip_files = glob(zip_pattern)
    
    # Ekstrakcja nazw zadań z nazw plików
    job_names = [os.path.splitext(os.path.basename(f))[0].replace("_results", "") for f in zip_files]
    logger.debug("Znalezione zadania: %s", job_names)
    return job_names

@router.get("/model_versions_maskrcnn")
def get_model_versions():
    """
    Endpoint do pobierania listy dostępnych wersji modeli Mask R-CNN.
    
    Zwraca listę nazw plików modeli, które mogą być użyte
    do automatycznego oznaczania obrazów.
    
    Returns:
        list: Lista nazw plików modeli Mask R-CNN
    """
    #######################
    # Pobieranie dostępnych modeli
    #######################
    versions = auto_label_api.get_model_versions()
    logger.debug("Dostępne wersje modeli Mask R-CNN: %s", versions)
    return versions