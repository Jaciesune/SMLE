import os
import logging
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from api.benchmark_api import BenchmarkAPI
import json

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()
benchmark_api = BenchmarkAPI()

@router.post("/run_benchmark")
async def run_benchmark(
    images: list[UploadFile] = File(...),
    annotations: list[UploadFile] = File(...),
    json_data: str = File(...),  # Dane JSON jako pole formularza
    request: Request = None
):
    user_role = request.headers.get("X-User-Role")
    logger.debug(f"[DEBUG] Wywołanie /run_benchmark: user_role={user_role}, liczba obrazów={len(images)}, liczba annotacji={len(annotations)}")
    if user_role != "admin":
        logger.error("[DEBUG] Brak uprawnień: użytkownik nie jest adminem")
        raise HTTPException(status_code=403, detail="Only admins can run benchmark")

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

    # Przekaż dane do metody w BenchmarkAPI
    result = await benchmark_api.prepare_and_run_benchmark(images, annotations, data)
    logger.debug(f"[DEBUG] Wynik benchmarku: {result}")
    return result

@router.get("/get_benchmark_results")
async def get_benchmark_results(request: Request):
    user_role = request.headers.get("X-User-Role")
    logger.debug(f"[DEBUG] Wywołanie /get_benchmark_results: user_role={user_role}")
    if user_role is None:
        logger.error("[DEBUG] Brak nagłówka X-User-Role")
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = benchmark_api.get_benchmark_results()
    logger.debug(f"[DEBUG] Wynik get_benchmark_results: {result}")
    return result

@router.get("/compare_models")
async def compare_models():
    logger.debug("[DEBUG] Wywołanie /compare_models")
    result = benchmark_api.compare_models()
    logger.debug(f"[DEBUG] Wynik compare_models: {result}")
    return result