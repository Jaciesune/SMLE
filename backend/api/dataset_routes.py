"""
Moduł tras API dla zarządzania zbiorami danych (Dataset Routes)

Ten moduł dostarcza endpointy FastAPI do tworzenia, zarządzania i pobierania zbiorów danych
dla modeli uczenia maszynowego. Umożliwia tworzenie nowych zbiorów, listowanie istniejących,
pobieranie informacji o nich oraz ich pobieranie i usuwanie.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na ścieżkach i plikach
import logging           # Do logowania informacji i błędów
from fastapi import (
    APIRouter,           # Klasa do tworzenia routerów w FastAPI
    HTTPException,       # Klasa do zgłaszania wyjątków HTTP
    UploadFile,          # Klasa do obsługi przesyłanych plików
    File,                # Funkcja do deklarowania parametrów plików
    Form                 # Funkcja do deklarowania parametrów formularza
)
from fastapi.responses import FileResponse  # Klasa do zwracania plików jako odpowiedzi HTTP
from api.dataset_api import DatasetAPI      # Import naszego API zarządzania datasetami

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#######################
# Inicjalizacja routera i API
#######################
router = APIRouter()     # Utworzenie routera FastAPI
dataset_api = DatasetAPI()  # Inicjalizacja naszego API zarządzania datasetami

#######################
# Endpointy API
#######################

@router.post("/create_dataset")
async def create_dataset(
    username: str = Form(...),
    job_name: str = Form(...),
    train_ratio: str = Form(...),  # Form(...) z typem str
    val_ratio: str = Form(...),    # Form(...) z typem str
    test_ratio: str = Form(...),   # Form(...) z typem str
    files: list[UploadFile] = File(...)
):
    """
    Endpoint do tworzenia nowego zbioru danych.
    
    Przyjmuje pliki obrazów i adnotacji, a następnie tworzy zbiór danych 
    z podziałem na podzbiory treningowy, walidacyjny i testowy.
    
    Parameters:
        username (str): Nazwa użytkownika
        job_name (str): Nazwa zbioru danych
        train_ratio (str): Proporcja zbioru treningowego (jako string, np. "0.7")
        val_ratio (str): Proporcja zbioru walidacyjnego (jako string, np. "0.2")
        test_ratio (str): Proporcja zbioru testowego (jako string, np. "0.1")
        files (list[UploadFile]): Lista przesłanych plików (obrazy i adnotacje)
        
    Returns:
        dict: Słownik z informacją o statusie operacji
        
    Raises:
        HTTPException: 
            - 422: Gdy proporcje nie mogą być przekonwertowane na liczby
            - 500: Gdy wystąpi inny błąd podczas tworzenia zbioru danych
    """
    #######################
    # Logowanie i walidacja parametrów
    #######################
    logger.debug("Otrzymane parametry: username=%s, job_name=%s, train_ratio=%s, val_ratio=%s, test_ratio=%s",
                 username, job_name, train_ratio, val_ratio, test_ratio)
    logger.info("Rozpoczynam tworzenie datasetu dla username=%s, job_name=%s, %d plików", username, job_name, len(files))
    
    #######################
    # Tworzenie zbioru danych
    #######################
    try:
        # Konwersja stringów na floaty
        train_ratio_float = float(train_ratio)
        val_ratio_float = float(val_ratio)
        test_ratio_float = float(test_ratio)
        
        # Wywołanie API do utworzenia zbioru danych
        dataset_api.create_dataset(username, job_name, files, train_ratio_float, val_ratio_float, test_ratio_float)
        
        logger.info("Dataset %s utworzony pomyślnie dla użytkownika %s", job_name, username)
        return {"status": "success"}
    except ValueError as e:
        # Obsługa błędów konwersji parametrów
        logger.error("Błąd konwersji proporcji na liczby: %s", str(e))
        raise HTTPException(status_code=422, detail=f"Błąd walidacji proporcji: {str(e)}")
    except Exception as e:
        # Obsługa innych błędów
        logger.error("Błąd w endpointcie /create_dataset: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia datasetu: {str(e)}")

@router.get("/list_datasets/{username}")
async def list_datasets(username: str):
    """
    Endpoint do pobierania listy zbiorów danych użytkownika.
    
    Parameters:
        username (str): Nazwa użytkownika
        
    Returns:
        dict: Słownik zawierający listę zbiorów danych w kluczu "datasets"
        
    Raises:
        HTTPException: 
            - 500: Gdy wystąpi błąd podczas pobierania listy zbiorów danych
    """
    #######################
    # Pobieranie listy zbiorów danych
    #######################
    try:
        # Wywołanie API do pobrania listy datasetów
        datasets = dataset_api.list_datasets(username)
        
        logger.info("Pobrano listę datasetów dla użytkownika %s: %s", username, datasets)
        return {"datasets": datasets}
    except Exception as e:
        # Obsługa błędów
        logger.error("Błąd w endpointcie /list_datasets: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas pobierania listy datasetów: {str(e)}")

@router.get("/dataset_info/{username}/{dataset_name}")
async def dataset_info(username: str, dataset_name: str):
    """
    Endpoint do pobierania szczegółowych informacji o zbiorze danych.
    
    Zwraca informacje o liczbie i nazwach obrazów w każdym podzbiorze 
    (treningowym, walidacyjnym i testowym).
    
    Parameters:
        username (str): Nazwa użytkownika
        dataset_name (str): Nazwa zbioru danych
        
    Returns:
        dict: Słownik zawierający informacje o podzbiorach (train, val, test)
              z liczbą i nazwami obrazów w każdym
        
    Raises:
        HTTPException: 
            - 500: Gdy wystąpi błąd podczas pobierania informacji o zbiorze danych
    """
    #######################
    # Pobieranie informacji o zbiorze danych
    #######################
    try:
        # Wywołanie API do pobrania informacji o datasecie
        info = dataset_api.get_dataset_info(username, dataset_name)
        
        logger.info("Pobrano informacje o datasecie %s dla użytkownika %s", dataset_name, username)
        return info
    except Exception as e:
        # Obsługa błędów
        logger.error("Błąd w endpointcie /dataset_info: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas pobierania informacji o datasecie: {str(e)}")

@router.get("/download_dataset/{username}/{dataset_name}/{subset}")
async def download_dataset_subset(username: str, dataset_name: str, subset: str):
    """
    Endpoint do pobierania wybranego podzbioru danych jako pliku ZIP.
    
    Parameters:
        username (str): Nazwa użytkownika
        dataset_name (str): Nazwa zbioru danych
        subset (str): Nazwa podzbioru (train, val, test)
        
    Returns:
        FileResponse: Odpowiedź HTTP zawierająca plik ZIP z podbiorem danych
        
    Raises:
        HTTPException: 
            - 500: Gdy wystąpi błąd podczas przygotowywania pliku ZIP
    """
    #######################
    # Pobieranie podzbioru danych
    #######################
    try:
        # Wywołanie API do pobrania ścieżki do pliku ZIP z podbiorem
        zip_path = dataset_api.download_dataset(username, dataset_name, subset)
        
        logger.info("Pobrano podzbiór %s datasetu %s dla użytkownika %s", subset, dataset_name, username)
        # Zwróć plik ZIP jako odpowiedź HTTP
        return FileResponse(zip_path, filename=f"{dataset_name}_{subset}_results.zip")
    except Exception as e:
        # Obsługa błędów
        logger.error("Błąd w endpointcie /download_dataset (subset): %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")

@router.get("/download_dataset/{username}/{dataset_name}")
async def download_dataset_full(username: str, dataset_name: str):
    """
    Endpoint do pobierania pełnego zbioru danych jako pliku ZIP.
    
    Parameters:
        username (str): Nazwa użytkownika
        dataset_name (str): Nazwa zbioru danych
        
    Returns:
        FileResponse: Odpowiedź HTTP zawierająca plik ZIP z pełnym zbiorem danych
        
    Raises:
        HTTPException: 
            - 500: Gdy wystąpi błąd podczas przygotowywania pliku ZIP
    """
    #######################
    # Pobieranie pełnego zbioru danych
    #######################
    try:
        # Wywołanie API do pobrania ścieżki do pliku ZIP z całym datasetem
        zip_path = dataset_api.download_dataset(username, dataset_name)
        
        logger.info("Pobrano cały dataset %s dla użytkownika %s", dataset_name, username)
        # Zwróć plik ZIP jako odpowiedź HTTP
        return FileResponse(zip_path, filename=f"{dataset_name}_full_results.zip")
    except Exception as e:
        # Obsługa błędów
        logger.error("Błąd w endpointcie /download_dataset (full): %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas zwracania pliku: {str(e)}")

@router.delete("/delete_zip/{username}/{dataset_name}/{subset}")
async def delete_zip(username: str, dataset_name: str, subset: str):
    """
    Endpoint do usuwania pliku ZIP z podbiorem danych.
    
    Parameters:
        username (str): Nazwa użytkownika
        dataset_name (str): Nazwa zbioru danych
        subset (str): Nazwa podzbioru (train, val, test) lub "full"
        
    Returns:
        dict: Słownik z wiadomością potwierdzającą usunięcie pliku ZIP
        
    Raises:
        HTTPException: 
            - 500: Gdy wystąpi błąd podczas usuwania pliku ZIP
    """
    #######################
    # Usuwanie pliku ZIP z podbiorem danych
    #######################
    try:
        # Utwórz ścieżkę do pliku ZIP
        zip_path = os.path.join("/app/backend/data/dataset_create", username, dataset_name, f"{subset}_results.zip")
        
        # Usuń plik, jeśli istnieje
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("Usunięto plik ZIP: %s", zip_path)
            
        return {"message": "ZIP usunięty"}
    except Exception as e:
        # Obsługa błędów
        logger.error("Błąd podczas usuwania ZIP: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania ZIP: {str(e)}")

@router.delete("/delete_zip/{username}/{dataset_name}")
async def delete_full_zip(username: str, dataset_name: str):
    """
    Endpoint do usuwania pliku ZIP z pełnym zbiorem danych.
    
    Parameters:
        username (str): Nazwa użytkownika
        dataset_name (str): Nazwa zbioru danych
        
    Returns:
        dict: Słownik z wiadomością potwierdzającą usunięcie pliku ZIP
        
    Raises:
        HTTPException: 
            - 500: Gdy wystąpi błąd podczas usuwania pliku ZIP
    """
    #######################
    # Usuwanie pliku ZIP z pełnym zbiorem danych
    #######################
    try:
        # Utwórz ścieżkę do pliku ZIP
        zip_path = os.path.join("/app/backend/data/dataset_create", username, dataset_name, "full_results.zip")
        
        # Usuń plik, jeśli istnieje
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("Usunięto plik ZIP: %s", zip_path)
            
        return {"message": "ZIP usunięty"}
    except Exception as e:
        # Obsługa błędów
        logger.error("Błąd podczas usuwania ZIP: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania ZIP: {str(e)}")

@router.delete("/delete_dataset/{username}/{dataset_name}")
async def delete_dataset(username: str, dataset_name: str):
    """
    Endpoint do usuwania całego zbioru danych.
    
    Parameters:
        username (str): Nazwa użytkownika
        dataset_name (str): Nazwa zbioru danych do usunięcia
        
    Returns:
        dict: Słownik z informacją o statusie operacji
        
    Raises:
        HTTPException: 
            - 500: Gdy wystąpi błąd podczas usuwania zbioru danych
    """
    #######################
    # Usuwanie całego zbioru danych
    #######################
    try:
        # Wywołanie API do usunięcia całego datasetu
        dataset_api.delete_dataset(username, dataset_name)
        
        logger.info("Usunięto dataset %s dla użytkownika %s", dataset_name, username)
        return {"status": "success"}
    except Exception as e:
        # Obsługa błędów
        logger.error("Błąd w endpointcie /delete_dataset: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Błąd podczas usuwania datasetu: {str(e)}")