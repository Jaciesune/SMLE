"""
Moduł tras API dla detekcji obiektów (Detection Routes)

Ten moduł dostarcza endpointy FastAPI do wykonywania detekcji obiektów na obrazach,
pobierania listy dostępnych algorytmów oraz wersji modeli. Stanowi interfejs REST
do komunikacji z funkcjonalnością detekcji obiektów.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na systemie plików
import shutil            # Do kopiowania plików
import logging           # Do logowania informacji i błędów
from fastapi import (
    APIRouter,           # Klasa do tworzenia routerów w FastAPI
    HTTPException,       # Klasa do zgłaszania wyjątków HTTP
    UploadFile,          # Klasa do obsługi przesyłanych plików
    File,                # Funkcja do deklarowania parametrów plików
    Form                 # Funkcja do deklarowania parametrów formularza
)
from fastapi.responses import FileResponse  # Klasa do zwracania plików jako odpowiedzi HTTP
from api.detection_api import DetectionAPI  # Import naszego API detekcji
from archive_tab import get_db_connection   # Funkcja do łączenia z bazą danych archiwum

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("api.detection_routes")

#######################
# Inicjalizacja routera i API
#######################
router = APIRouter()       # Utworzenie routera FastAPI
detection_api = DetectionAPI()  # Inicjalizacja naszego API detekcji

#######################
# Endpointy API
#######################

@router.get("/detect_algorithms")
def get_algorithms():
    """
    Endpoint do pobierania listy dostępnych algorytmów detekcji.
    
    Returns:
        list: Lista nazw dostępnych algorytmów (np. Mask R-CNN, FasterRCNN, MCNN)
    """
    algs = detection_api.get_algorithms()
    logger.debug("Zwracam algorytmy: %s", algs)
    return algs

@router.get("/detect_model_versions/{algorithm}")
def get_model_versions(algorithm: str):
    """
    Endpoint do pobierania listy wersji modeli dla wybranego algorytmu.
    
    Parameters:
        algorithm (str): Nazwa algorytmu detekcji
        
    Returns:
        list: Lista nazw dostępnych modeli dla wybranego algorytmu
    """
    versions = detection_api.get_model_versions(algorithm)
    logger.debug("Zwracam wersje modeli dla %s: %s", algorithm, versions)
    return versions
    
@router.post("/detect_image")
async def detect_image(
    algorithm: str = Form(...),
    model_version: str = Form(...),
    image: UploadFile = File(...),
    username: str = Form(None),
    preprocessing: bool = Form(False)
):
    """
    Endpoint do przeprowadzania detekcji obiektów na przesłanym obrazie.
    
    Przyjmuje obraz, informacje o wybranym algorytmie i modelu, a następnie
    wykonuje detekcję obiektów i zwraca obraz z zaznaczonymi detekcjami.
    
    Parameters:
        algorithm (str): Nazwa algorytmu detekcji
        model_version (str): Nazwa pliku modelu
        image (UploadFile): Przesłany plik obrazu
        username (str, optional): Nazwa użytkownika (do zapisu w archiwum). Domyślnie None.
        preprocessing (bool, optional): Czy zastosować preprocessing obrazu. Domyślnie False.
        
    Returns:
        FileResponse: Odpowiedź HTTP zawierająca plik z obrazem z zaznaczonymi detekcjami
                     oraz nagłówek X-Detections-Count z liczbą wykrytych obiektów
        
    Raises:
        HTTPException: 
            - 400: Gdy podano nieobsługiwany algorytm
            - 500: Gdy wystąpił błąd podczas detekcji lub gdy nie udało się zapisać wyniku
    """
    #######################
    # Logowanie parametrów wejściowych
    #######################
    logger.debug(
        "Rozpoczynam detekcję: algorithm=%s, model_version=%s, image=%s, username=%s, preprocessing=%s",
        algorithm, model_version, image.filename, username, preprocessing
    )

    #######################
    # Wybór ścieżek dla algorytmów
    #######################
    if algorithm == "Mask R-CNN":
        input_dir = "/app/backend/Mask_RCNN/data/test/images"
        output_dir = "/app/backend/Mask_RCNN/data/detectes"
    elif algorithm == "FasterRCNN":
        input_dir = "/app/backend/FasterRCNN/data/test/images"
        output_dir = "/app/backend/FasterRCNN/data/detectes"
    elif algorithm == "MCNN":
        input_dir = "/app/backend/MCNN/data/test/images"
        output_dir = "/app/backend/MCNN/data/detectes"
    else:
        logger.error("Nieobsługiwany algorytm: %s", algorithm)
        raise HTTPException(400, f"Nieobsługiwany algorytm: {algorithm}")

    #######################
    # Przygotowanie i detekcja obrazu
    #######################
    try:
        # Utworzenie katalogów, jeśli nie istnieją
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Zapisanie przesłanego obrazu do pliku
        img_path = os.path.join(input_dir, image.filename)
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        logger.debug("Zapisano zdjęcie: %s", img_path)

        #######################
        # Uruchomienie detekcji
        #######################
        # Uruchomienie detekcji z preprocessingiem (jeśli włączone)
        result_path, count = detection_api.analyze_with_model(
            img_path, algorithm, model_version, preprocessing  # Przekazujemy parametr preprocessing
        )
        logger.debug("Wynik detekcji: result=%s, count=%d", result_path, count)

        # Sprawdzenie błędów w wyniku detekcji
        if "Błąd" in result_path or not os.path.exists(result_path):
            logger.error("Detekcja nie powiodła się: %s", result_path)
            raise HTTPException(500, f"Detekcja nie powiodła się albo wynik nie istnieje: {result_path}")

        #######################
        # Zapis do archiwum (opcjonalnie)
        #######################
        if username:
            try:
                # Połączenie z bazą danych
                conn = get_db_connection()
                cur = conn.cursor()
                
                # Pobranie ID użytkownika
                cur.execute("SELECT id FROM user WHERE name = %s", (username,))
                row = cur.fetchone()
                if row:
                    user_id = row[0]
                else:
                    raise Exception(f"Nie znaleziono użytkownika: {username}")
                
                # Wyciągnij nazwę modelu przed pierwszym podkreśleniem
                short_model_name = model_version.split('_')[0]

                # Pobranie ID modelu
                cur.execute("SELECT id FROM model WHERE name = %s AND algorithm = %s", (short_model_name, algorithm))
                model_row = cur.fetchone()
                if model_row:
                    model_id = model_row[0]
                else:
                    raise Exception(f"Nie znaleziono modelu: {short_model_name} dla algorytmu: {algorithm}")

                # Zapis akcji detekcji do archiwum
                cur.execute(
                    "INSERT INTO archive(action, user_id, model_id, date) VALUES (%s,%s,%s,NOW())",
                    ("Detekcja obrazu", user_id, model_id)
                )

                # Zatwierdzenie transakcji i zamknięcie połączenia
                conn.commit()
                cur.close()
                conn.close()
                logger.debug("Zapisano do archive: user_id=%s, model_id=%s", user_id, model_id)
            except Exception as e:
                # Obsługa błędów zapisu do archiwum (nie przerywamy działania endpointu)
                logger.error("Nie udało się zapisać do archiwum: %s", e)
        else:
            logger.warning("Nie podano username, pomijam zapis do archiwum.")

        #######################
        # Przygotowanie odpowiedzi
        #######################
        # Utworzenie odpowiedzi z plikiem obrazu wynikowego
        resp = FileResponse(result_path, filename=os.path.basename(result_path))
        # Dodanie nagłówka z liczbą wykrytych obiektów
        resp.headers["X-Detections-Count"] = str(count)
        return resp

    except HTTPException:
        # Przepuść wyjątki HTTPException dalej
        raise
    except Exception as e:
        # Obsługa innych błędów
        logger.exception("Błąd w endpointcie /detect_image")
        raise HTTPException(500, str(e))
    finally:
        # Zamknięcie pliku obrazu (wymagane przez FastAPI)
        await image.close()