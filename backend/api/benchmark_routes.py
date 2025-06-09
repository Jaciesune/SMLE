"""
Moduł tras API dla testów porównawczych (Benchmark Routes)

Ten moduł dostarcza endpointy FastAPI do przeprowadzania testów porównawczych modeli,
pobierania wyników benchmarków oraz porównywania skuteczności różnych modeli.
Stanowi interfejs REST do komunikacji z funkcjonalnością benchmarkingu.
"""

#######################
# Importy bibliotek
#######################
import os                   # Do operacji na systemie plików
import logging              # Do logowania informacji i błędów
import json                 # Do parsowania danych JSON
from fastapi import (
    APIRouter,              # Klasa do tworzenia routerów w FastAPI
    HTTPException,          # Klasa do zgłaszania wyjątków HTTP
    Request,                # Klasa reprezentująca żądanie HTTP
    UploadFile,             # Klasa do obsługi przesyłanych plików
    File                    # Funkcja do deklarowania parametrów plików
)
from api.benchmark_api import BenchmarkAPI  # Import naszego API benchmarkingu

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#######################
# Inicjalizacja routera i API
#######################
router = APIRouter()  # Utworzenie routera FastAPI
benchmark_api = BenchmarkAPI()  # Inicjalizacja naszego API benchmarkingu

#######################
# Endpointy API
#######################

@router.post("/run_benchmark")
async def run_benchmark(
    images: list[UploadFile] = File(...),
    annotations: list[UploadFile] = File(...),
    json_data: str = File(...),  # Dane JSON jako pole formularza
    request: Request = None
):
    """
    Endpoint do uruchamiania testu porównawczego modelu.
    
    Przyjmuje obrazy testowe i ich annotacje, parametry benchmarku w formie JSON,
    a następnie przeprowadza test porównawczy wybranego modelu.
    
    Parameters:
        images (list[UploadFile]): Lista plików obrazów testowych
        annotations (list[UploadFile]): Lista plików annotacji odpowiadających obrazom
        json_data (str): Dane JSON zawierające parametry benchmarku (algorithm, model_version, model_name, source_folder)
        request (Request): Obiekt żądania HTTP zawierający nagłówki
        
    Returns:
        dict: Wyniki benchmarku zawierające metryki:
            - MAE: Średni błąd bezwzględny (Mean Absolute Error)
            - effectiveness: Skuteczność modelu w procentach
            - algorithm: Nazwa algorytmu
            - model_version: Wersja modelu
            - model_name: Nazwa modelu
            - image_folder: Folder testowy z obrazami
            - source_folder: Folder źródłowy danych
            - timestamp: Znacznik czasu wykonania benchmarku
            
    Raises:
        HTTPException: 
            - 403: Gdy użytkownik nie ma uprawnień administratora
            - 400: Gdy liczba obrazów i annotacji nie jest zgodna lub dane JSON są nieprawidłowe
    """
    #######################
    # Sprawdzenie uprawnień
    #######################
    # Pobierz rolę użytkownika z nagłówka
    user_role = request.headers.get("X-User-Role")
    logger.debug(f"[DEBUG] Wywołanie /run_benchmark: user_role={user_role}, liczba obrazów={len(images)}, liczba annotacji={len(annotations)}")
    
    # Sprawdź, czy użytkownik ma uprawnienia administratora
    if user_role != "admin":
        logger.error("[DEBUG] Brak uprawnień: użytkownik nie jest adminem")
        raise HTTPException(status_code=403, detail="Only admins can run benchmark")

    #######################
    # Walidacja danych wejściowych
    #######################
    # Sprawdź, czy liczba obrazów i annotacji jest zgodna
    if len(images) != len(annotations):
        logger.error(f"[DEBUG] Niezgodna liczba obrazów ({len(images)}) i annotacji ({len(annotations)})")
        raise HTTPException(status_code=400, detail="Number of images and annotations must match")

    # Parsuj dane JSON z pola formularza
    try:
        data = json.loads(json_data)
        logger.debug(f"[DEBUG] Dane JSON z żądania: {data}")
    except json.JSONDecodeError as e:
        logger.error(f"[DEBUG] Błąd parsowania JSON: {e}")
        raise HTTPException(status_code=400, detail="Nieprawidłowy format danych JSON")

    #######################
    # Uruchomienie benchmarku
    #######################
    # Przekaż dane do metody w BenchmarkAPI
    result = await benchmark_api.prepare_and_run_benchmark(images, annotations, data)
    logger.debug(f"[DEBUG] Wynik benchmarku: {result}")
    
    # Zwróć wyniki benchmarku
    return result

@router.get("/get_benchmark_results")
async def get_benchmark_results(request: Request):
    """
    Endpoint do pobierania historii wyników testów porównawczych.
    
    Zwraca historię wszystkich przeprowadzonych benchmarków dla różnych modeli.
    Wyniki są pogrupowane chronologicznie, zawierając wszystkie metryki i metadane.
    
    Parameters:
        request (Request): Obiekt żądania HTTP zawierający nagłówki
        
    Returns:
        dict: Słownik zawierający historię wyników benchmarków w kluczu "history":
            - history: Lista wszystkich zapisanych wyników benchmarków z metrykami i metadanymi
            
    Raises:
        HTTPException: 
            - 401: Gdy brak nagłówka uwierzytelniającego
            - 404: Gdy nie ma dostępnych wyników benchmarków
            - 500: Gdy wystąpił błąd podczas odczytu wyników
    """
    #######################
    # Sprawdzenie uwierzytelnienia
    #######################
    # Pobierz rolę użytkownika z nagłówka
    user_role = request.headers.get("X-User-Role")
    logger.debug(f"[DEBUG] Wywołanie /get_benchmark_results: user_role={user_role}")
    
    # Sprawdź, czy nagłówek X-User-Role istnieje
    if user_role is None:
        logger.error("[DEBUG] Brak nagłówka X-User-Role")
        raise HTTPException(status_code=401, detail="Unauthorized")

    #######################
    # Pobieranie wyników
    #######################
    # Pobierz historię wyników benchmarków
    result = benchmark_api.get_benchmark_results()
    logger.debug(f"[DEBUG] Wynik get_benchmark_results: {result}")
    
    # Zwróć historię wyników
    return result

@router.get("/compare_models")
async def compare_models():
    """
    Endpoint do porównywania skuteczności różnych modeli.
    
    Analizuje historię benchmarków, grupuje wyniki według zbiorów danych
    i wybiera najlepsze modele dla każdego zbioru oraz najlepszy model ogółem.
    Wyniki są sortowane według skuteczności modeli.
    
    Returns:
        dict: Słownik zawierający:
            - results: Lista wyników porównań zgrupowanych według zbiorów danych, 
              gdzie każdy element zawiera:
                - dataset: Nazwa zbioru danych
                - results: Lista wyników wszystkich modeli dla danego zbioru
                - best_model: Informacje o najlepszym modelu dla danego zbioru
            - best_model: Informacje o najlepszym modelu ogółem:
                - dataset: Nazwa zbioru danych
                - model: Pełna nazwa modelu (algorytm i wersja)
                - model_name: Bazowa nazwa modelu
                - effectiveness: Skuteczność modelu w procentach
    """
    #######################
    # Porównywanie modeli
    #######################
    # Wywołaj metodę porównującą modele w BenchmarkAPI
    logger.debug("[DEBUG] Wywołanie /compare_models")
    result = benchmark_api.compare_models()
    logger.debug(f"[DEBUG] Wynik compare_models: {result}")
    
    # Zwróć wyniki porównania
    return result