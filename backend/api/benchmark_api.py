"""
Moduł API testów porównawczych (Benchmark API)

Ten moduł dostarcza interfejs do przeprowadzania testów porównawczych (benchmarków)
dla modeli detekcji obiektów. Umożliwia ewaluację skuteczności modeli na zbiorze
testowym obrazów z annotacjami oraz porównywanie wyników różnych modeli.
"""

#######################
# Importy bibliotek
#######################
import os                   # Do operacji na ścieżkach i plikach
import shutil               # Do kopiowania i usuwania plików/katalogów
import json                 # Do operacji na plikach JSON
import logging              # Do logowania informacji i błędów
from datetime import datetime  # Do dodawania znaczników czasu do wyników
from fastapi import HTTPException  # Do zgłaszania wyjątków HTTP
from api.detection_api import DetectionAPI  # Interfejs do modeli detekcji

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#######################
# Klasa zarządzania danymi obrazów
#######################
class ImageDataset:
    """
    Klasa do zarządzania zbiorem danych obrazów i ich annotacji.
    
    Odpowiada za ładowanie obrazów oraz ich odpowiadających annotacji w formacie LabelMe,
    udostępniając interfejs indeksowania do dostępu do par (obraz, annotacja).
    """
    
    def __init__(self, image_folder, annotation_path):
        """
        Inicjalizacja zbioru danych obrazów.
        
        Parameters:
            image_folder (str): Ścieżka do folderu z obrazami
            annotation_path (str): Ścieżka do folderu z plikami annotacji
        """
        # Zapisz ścieżki do folderów
        self.image_folder = image_folder
        self.annotation_path = annotation_path
        logger.debug(f"[DEBUG] Inicjalizacja ImageDataset: image_folder={self.image_folder}, annotation_path={self.annotation_path}")
        
        # Wczytaj listę obrazów (tylko pliki o rozszerzeniach .jpg i .png)
        self.images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
        logger.debug(f"[DEBUG] Znalezione obrazy: {self.images}")
        if not self.images:
            logger.warning(f"[DEBUG] Brak obrazów w folderze {self.image_folder}")

        # Wczytaj annotacje w formacie LabelMe dla każdego obrazu
        self.file_to_annotations = {}
        for img_file in self.images:
            # Szukaj pliku JSON o nazwie odpowiadającej nazwie obrazu
            annotation_file = os.path.join(self.annotation_path, f"{os.path.splitext(img_file)[0]}.json")
            logger.debug(f"[DEBUG] Szukam annotacji dla {img_file}: {annotation_file}")
            
            # Jeśli plik istnieje, wczytaj jego zawartość
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    ann_data = json.load(f)
                # Zapisz listę kształtów (obiektów) z annotacji
                self.file_to_annotations[img_file] = ann_data.get("shapes", [])
                logger.debug(f"[DEBUG] Znaleziono annotacje dla {img_file}: {len(self.file_to_annotations[img_file])} kształtów")
            else:
                # Jeśli nie ma annotacji, zapisz pustą listę
                logger.warning(f"[DEBUG] Brak pliku annotacji dla obrazu {img_file}")
                self.file_to_annotations[img_file] = []

    def __len__(self):
        """
        Zwraca liczbę obrazów w zbiorze danych.
        
        Returns:
            int: Liczba obrazów w zbiorze danych
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Pobiera parę (ścieżka do obrazu, annotacje) dla podanego indeksu.
        
        Metoda umożliwia dostęp do zbioru danych poprzez indeksowanie, np. dataset[0].
        
        Parameters:
            idx (int): Indeks obrazu w zbiorze danych
            
        Returns:
            tuple: Para (ścieżka do obrazu, annotacje)
        """
        # Pobierz nazwę pliku dla danego indeksu
        file_name = self.images[idx]
        # Utwórz pełną ścieżkę do pliku
        img_path = os.path.join(self.image_folder, file_name)
        # Pobierz annotacje dla pliku (lub pustą listę, jeśli nie istnieją)
        annotations = self.file_to_annotations.get(file_name, [])
        # Zwróć parę (ścieżka, annotacje)
        return img_path, annotations

#######################
# Główna klasa API benchmarku
#######################
class BenchmarkAPI:
    """
    API do przeprowadzania testów porównawczych modeli detekcji obiektów.
    
    Umożliwia przygotowanie danych testowych, przeprowadzenie benchmarku na wybranych modelach,
    zapisywanie wyników oraz porównywanie skuteczności różnych modeli.
    """
    
    def __init__(self):
        """
        Inicjalizacja BenchmarkAPI z domyślnymi ścieżkami.
        
        Konfiguruje ścieżki do katalogów testowych i pliku historii wyników benchmarków.
        Tworzy niezbędne katalogi, jeśli nie istnieją.
        """
        # Inicjalizacja API detekcji do przeprowadzania detekcji na obrazach
        self.detection_api = DetectionAPI()
        
        # Ścieżki do katalogów dla testów benchmarkingowych
        self.image_folder = "/app/backend/data/test/images"            # Katalog obrazów testowych
        self.annotation_path = "/app/backend/data/test/annotations"    # Katalog annotacji
        self.history_file = "/app/backend/benchmark_history.json"      # Plik z historią benchmarków
        logger.debug("[DEBUG] Inicjalizacja BenchmarkAPI")

        # Tworzenie potrzebnych katalogów przy starcie
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotation_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        logger.debug(f"[DEBUG] Utworzono foldery przy inicjalizacji: images={self.image_folder}, annotations={self.annotation_path}")

    async def prepare_and_run_benchmark(self, images, annotations, request):
        """
        Przygotowuje dane testowe i wykonuje benchmark wybranego modelu.
        
        Metoda ta łączy dwa kroki: przygotowanie danych (obrazów i annotacji)
        oraz przeprowadzenie testu porównawczego na wybranym modelu.
        
        Parameters:
            images (list[UploadFile]): Lista plików obrazów do przetestowania
            annotations (list[UploadFile]): Lista plików annotacji dla obrazów
            request (dict): Słownik zawierający parametry benchmarku:
                - algorithm (str): Nazwa algorytmu do wykorzystania (np. "Mask R-CNN", "MCNN")
                - model_version (str): Wersja modelu
                - model_name (str): Nazwa modelu
                - source_folder (str): Nazwa folderu źródłowego danych (opcjonalne)
                
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
            HTTPException: Gdy wystąpi błąd podczas przygotowania lub wykonania benchmarku
        """
        #######################
        # Krok 1: Przygotowanie danych testowych
        #######################
        logger.debug(f"[DEBUG] Przygotowywanie danych: liczba obrazów={len(images)}, liczba annotacji={len(annotations)}")
        
        # Wyczyść stare dane (usuń katalogi jeśli istnieją)
        if os.path.exists(self.image_folder):
            logger.debug(f"[DEBUG] Usuwanie starego folderu obrazów: {self.image_folder}")
            shutil.rmtree(self.image_folder)
        if os.path.exists(self.annotation_path):
            logger.debug(f"[DEBUG] Usuwanie starego folderu annotacji: {self.annotation_path}")
            shutil.rmtree(self.annotation_path)

        # Stwórz nowe katalogi
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotation_path, exist_ok=True)
        logger.debug(f"[DEBUG] Utworzono nowe foldery: images={self.image_folder}, annotations={self.annotation_path}")

        # Zapisywanie przesłanych obrazów do katalogu
        for img_file in images:
            img_path = os.path.join(self.image_folder, img_file.filename)
            with open(img_path, "wb") as f:
                content = await img_file.read()
                if not content:
                    logger.error(f"[DEBUG] Obraz {img_file.filename} jest pusty!")
                    raise HTTPException(status_code=400, detail=f"Obraz {img_file.filename} jest pusty")
                f.write(content)
            logger.debug(f"[DEBUG] Zapisano obraz: {img_path}, rozmiar={os.path.getsize(img_path)} bajtów")

        # Zapisywanie przesłanych annotacji do katalogu
        for ann_file in annotations:
            ann_path = os.path.join(self.annotation_path, ann_file.filename)
            with open(ann_path, "wb") as f:
                content = await ann_file.read()
                if not content:
                    logger.error(f"[DEBUG] Annotacja {ann_file.filename} jest pusta!")
                    raise HTTPException(status_code=400, detail=f"Annotacja {ann_file.filename} jest pusta")
                f.write(content)
            logger.debug(f"[DEBUG] Zapisano annotację: {ann_path}, rozmiar={os.path.getsize(ann_path)} bajtów")

        # Sprawdzenie, czy katalogi nie są puste po zapisie
        images_list = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        annotations_list = [f for f in os.listdir(self.annotation_path) if f.endswith('.json')]
        logger.debug(f"[DEBUG] Po zapisie: obrazy={images_list}, annotacje={annotations_list}")
        
        # Walidacja danych testowych
        if not images_list:
            logger.error("[DEBUG] Brak obrazów w folderze po zapisie!")
            raise HTTPException(status_code=500, detail="Brak obrazów w folderze po zapisie")
        if not annotations_list:
            logger.error("[DEBUG] Brak annotacji w folderze po zapisie!")
            raise HTTPException(status_code=500, detail="Brak annotacji w folderze po zapisie")

        #######################
        # Krok 2: Uruchomienie benchmarku
        #######################
        # Pobierz parametry z żądania
        algorithm = request.get("algorithm")              # Nazwa algorytmu (Mask R-CNN, MCNN, FasterRCNN)
        model_version = request.get("model_version")      # Wersja modelu
        model_name = request.get("model_name")            # Nazwa modelu
        source_folder = request.get("source_folder", "")  # Folder źródłowy danych (opcjonalny)

        logger.debug(f"[DEBUG] Uruchamianie benchmarku: algorithm={algorithm}, model_version={model_version}, model_name={model_name}, source_folder={source_folder}")

        # Ponowne sprawdzenie istnienia katalogów testowych (na wszelki wypadek)
        if not os.path.exists(self.image_folder):
            logger.error(f"[DEBUG] Folder z obrazami nie istnieje: {self.image_folder}")
            raise HTTPException(status_code=400, detail=f"Folder z obrazami nie istnieje: {self.image_folder}")
        if not os.path.exists(self.annotation_path):
            logger.error(f"[DEBUG] Folder z annotacjami nie istnieje: {self.annotation_path}")
            raise HTTPException(status_code=400, detail=f"Folder z annotacjami nie istnieje: {self.annotation_path}")

        # Sprawdzenie zawartości katalogów
        images_list = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        annotations_list = [f for f in os.listdir(self.annotation_path) if f.endswith('.json')]
        logger.debug(f"[DEBUG] Przed benchmarkiem: obrazy={images_list}, annotacje={annotations_list}")
        
        # Walidacja zawartości katalogów
        if not images_list:
            logger.error(f"[DEBUG] Folder z obrazami jest pusty: {self.image_folder}")
            raise HTTPException(status_code=400, detail=f"Folder z obrazami jest pusty: {self.image_folder}")
        if not annotations_list:
            logger.error(f"[DEBUG] Folder z annotacjami jest pusty: {self.annotation_path}")
            raise HTTPException(status_code=400, detail=f"Folder z annotacjami jest pusty: {self.annotation_path}")

        # Wczytaj dane za pomocą klasy ImageDataset
        dataset = ImageDataset(self.image_folder, self.annotation_path)
        if len(dataset) == 0:
            logger.error("[DEBUG] Brak obrazów w podanym folderze")
            raise HTTPException(status_code=400, detail="Brak obrazów w podanym folderze")

        #######################
        # Krok 3: Konfiguracja ścieżek dla algorytmów
        #######################
        # Mapowanie katalogów obrazów dla każdego algorytmu w kontenerze Docker
        algorithm_to_path = {
            "Mask R-CNN": "/app/backend/Mask_RCNN/data/test/images",
            "MCNN": "/app/backend/MCNN/data/test/images",
            "FasterRCNN": "/app/backend/FasterRCNN/data/test/images",
        }
        if algorithm not in algorithm_to_path:
            logger.error(f"[DEBUG] Algorytm {algorithm} nie jest obsługiwany")
            raise HTTPException(status_code=400, detail=f"Algorytm {algorithm} nie jest obsługiwany")

        # Sprawdzenie, czy model istnieje - mapowanie katalogów z modelami
        model_paths = {
            "Mask R-CNN": "/app/backend/Mask_RCNN/models/",
            "MCNN": "/app/backend/MCNN/models/",
            "FasterRCNN": "/app/backend/FasterRCNN/saved_models/",
        }
        file_name = f"{model_name}_checkpoint.pth"  # Format nazwy pliku modelu
        model_path = os.path.join(model_paths.get(algorithm, ""), file_name)
        
        # Walidacja istnienia pliku modelu
        if not os.path.exists(model_path):
            logger.error(f"[DEBUG] Plik modelu nie istnieje: {model_path}")
            raise HTTPException(status_code=400, detail=f"Plik modelu nie istnieje: {model_path}")

        # Tworzenie katalogu na obrazy w kontenerze, jeśli nie istnieje
        container_images_path = algorithm_to_path[algorithm]
        os.makedirs(container_images_path, exist_ok=True)
        logger.debug(f"[DEBUG] Utworzono folder dla algorytmu: {container_images_path}")

        #######################
        # Krok 4: Przetwarzanie obrazów i zliczanie metryk
        #######################
        metrics_list = []          # Lista różnic między liczbą obiektów wykrytych a rzeczywistych
        ground_truth_counts = []   # Lista liczb obiektów w annotacjach
        predicted_counts = []      # Lista liczb obiektów wykrytych przez model
        
        # Przetwarzanie każdego obrazu z datasetu
        for idx in range(len(dataset)):
            img_path, annotations = dataset[idx]
            logger.debug(f"[DEBUG] Przetwarzanie obrazu {idx+1}/{len(dataset)}: {img_path}")

            # Kopiowanie obrazu do katalogu właściwego dla algorytmu
            container_image_path = os.path.join(container_images_path, os.path.basename(img_path))
            logger.debug(f"[DEBUG] Kopiowanie obrazu z {img_path} do {container_image_path}")
            shutil.copy(img_path, container_image_path)

            try:
                # Uruchom detekcję na obrazie
                logger.debug(f"[DEBUG] Uruchamianie detekcji na obrazie {os.path.basename(img_path)}")
                result, num_predicted = self.detection_api.analyze_with_model(container_image_path, algorithm, file_name)
                
                # Sprawdź, czy detekcja się powiodła
                if "Błąd" in result:
                    logger.error(f"[DEBUG] Błąd detekcji dla {img_path}: {result}")
                    raise HTTPException(status_code=500, detail=f"Błąd detekcji dla {img_path}: {result}")
                logger.debug(f"[DEBUG] Liczba wykrytych rur: {num_predicted} dla {img_path}")

                # Policz obiekty z annotacji (format LabelMe)
                num_ground_truth = len(annotations)
                ground_truth_counts.append(num_ground_truth)
                predicted_counts.append(num_predicted)
                logger.debug(f"[DEBUG] Liczba rur w annotacji: {num_ground_truth} dla {img_path}")

                # Oblicz metrykę dla tego obrazu (różnica między liczbą wykrytą a rzeczywistą)
                metric = abs(num_predicted - num_ground_truth)
                metrics_list.append(metric)
                logger.debug(f"[DEBUG] Metryka dla {img_path}: {metric}")
            finally:
                # Usuń skopiowany obraz po przetworzeniu
                if os.path.exists(container_image_path):
                    os.unlink(container_image_path)
                    logger.debug(f"[DEBUG] Usunięto skopiowany obraz: {container_image_path}")

        #######################
        # Krok 5: Obliczenie wyników i zapis do pliku
        #######################
        if metrics_list:
            # Oblicz Mean Absolute Error (MAE) - średni błąd bezwzględny
            mae = sum(metrics_list) / len(metrics_list)
            
            # Oblicz średnią liczbę obiektów w ground truth
            avg_ground_truth = sum(ground_truth_counts) / len(ground_truth_counts) if ground_truth_counts else 1
            logger.debug(f"[DEBUG] Średnia liczba obiektów w ground truth: {avg_ground_truth}")
            
            # Oblicz średnią liczbę przewidywanych obiektów
            avg_predicted = sum(predicted_counts) / len(predicted_counts) if predicted_counts else 0
            logger.debug(f"[DEBUG] Średnia liczba przewidywanych obiektów: {avg_predicted}")
            
            # Oblicz skuteczność w procentach (im mniejszy MAE względem średniej liczby obiektów, tym lepiej)
            effectiveness = max(0, (1 - mae / avg_ground_truth)) * 100 if avg_ground_truth > 0 else 0
            
            # Przygotuj słownik z wynikami
            results = {
                "MAE": mae,
                "effectiveness": round(effectiveness, 2),  # Zaokrąglenie do 2 miejsc po przecinku
                "algorithm": algorithm,
                "model_version": model_version,
                "model_name": model_name,
                "image_folder": self.image_folder,
                "source_folder": source_folder,
                "timestamp": datetime.now().isoformat()    # Dodaj znacznik czasu
            }
            logger.debug(f"[DEBUG] Wynik benchmarku: MAE={mae}, Skuteczność={effectiveness}%, avg_ground_truth={avg_ground_truth}, avg_predicted={avg_predicted}")

            #######################
            # Krok 6: Zapisywanie wyników
            #######################
            # Zapisz wyniki do pliku benchmark_results.json
            try:
                with open("/app/backend/benchmark_results.json", "w") as f:
                    json.dump(results, f)
                logger.debug("[DEBUG] Wyniki zapisane do pliku benchmark_results.json")
            except Exception as e:
                logger.error(f"[DEBUG] Błąd zapisu wyników: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd zapisu wyników: {e}")

            # Zapisz wyniki do historii benchmarków
            try:
                # Wczytaj istniejącą historię, jeśli istnieje
                history = []
                if os.path.exists(self.history_file):
                    try:
                        with open(self.history_file, "r") as f:
                            history = json.load(f)
                            # Sprawdź, czy wczytana historia jest listą
                            if not isinstance(history, list):
                                history = []
                    except json.JSONDecodeError:
                        history = []

                # Sprawdź, czy istnieje już wpis dla tego modelu i danych
                found = False
                for i, entry in enumerate(history):
                    if (entry.get("model_name") == model_name and
                        entry.get("algorithm") == algorithm and
                        entry.get("model_version") == model_version and
                        entry.get("source_folder") == source_folder):
                        # Jeśli istnieje, nadpisz go nowymi wynikami
                        history[i] = results
                        found = True
                        logger.debug(f"[DEBUG] Nadpisz istniejący wpis w historii dla modelu {model_name}")
                        break

                # Jeśli nie znaleziono istniejącego wpisu, dodaj nowy
                if not found:
                    history.append(results)
                    logger.debug(f"[DEBUG] Dodano nowy wpis do historii dla modelu {model_name}")

                # Zapisz zaktualizowaną historię do pliku
                with open(self.history_file, "w") as f:
                    json.dump(history, f, indent=4)  # Zapisz z wcięciami dla czytelności
                logger.debug("[DEBUG] Zaktualizowano historię benchmarków")
            except Exception as e:
                logger.error(f"[DEBUG] Błąd zapisu historii benchmarków: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd zapisu historii: {e}")

            return results
        else:
            # Brak przetworzonych obrazów
            logger.error("[DEBUG] Brak przetworzonych par obraz-annotacja")
            raise HTTPException(status_code=400, detail="No valid image-annotation pairs processed")

    def get_benchmark_results(self):
        """
        Pobiera historię wyników benchmarków.
        
        Odczytuje plik historii benchmarków i zwraca wszystkie zapisane wyniki.
        
        Returns:
            dict: Słownik zawierający historię wyników benchmarków w kluczu "history"
            
        Raises:
            HTTPException: Gdy nie ma dostępnych wyników lub wystąpił błąd podczas ich odczytu
        """
        try:
            # Odczytaj plik historii, jeśli istnieje
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    history = json.load(f)
                    # Sprawdź poprawność formatu
                    if not isinstance(history, list):
                        history = []
            else:
                history = []
            logger.debug(f"[DEBUG] Zwrócono historię: {history}")
            return {"history": history}
        except FileNotFoundError:
            logger.error("[DEBUG] Brak pliku z wynikami benchmarku")
            raise HTTPException(status_code=404, detail="No benchmark results available")
        except Exception as e:
            logger.error(f"[DEBUG] Błąd podczas odczytu wyników: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd podczas odczytu wyników: {e}")

    def compare_models(self):
        """
        Porównuje wyniki różnych modeli na podstawie historii benchmarków.
        
        Grupuje wyniki według zbiorów danych i wybiera najlepszy model dla każdego zbioru
        oraz najlepszy model ogółem.
        
        Returns:
            dict: Słownik zawierający:
                - results: Lista wyników porównań zgrupowanych według zbiorów danych
                - best_model: Informacje o najlepszym modelu ogółem
        """
        # Sprawdź, czy plik historii istnieje
        if not os.path.exists(self.history_file):
            logger.debug("[DEBUG] Brak historii benchmarków")
            return {"results": [], "best_model": None}

        try:
            # Wczytaj historię benchmarków
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"[DEBUG] Błąd odczytu historii benchmarków: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd odczytu historii: {e}")

        # Jeśli historia jest pusta, zwróć puste wyniki
        if not history:
            logger.debug("[DEBUG] Historia benchmarków jest pusta")
            return {"results": [], "best_model": None}

        # Grupowanie wyników według zbiorów danych (source_folder)
        results_by_dataset = {}
        for result in history:
            dataset = result.get("source_folder", "unknown_dataset")
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = []
            results_by_dataset[dataset].append(result)

        # Przygotowanie wyników porównania
        comparison_results = []  # Lista wyników dla każdego zbioru danych
        best_model_info = None   # Informacje o najlepszym modelu ogółem
        overall_best_effectiveness = -1  # Najlepsza skuteczność ogółem

        # Analiza wyników dla każdego zbioru danych
        for dataset, results in results_by_dataset.items():
            dataset_results = []  # Lista wyników dla bieżącego zbioru danych
            best_effectiveness = -1  # Najlepsza skuteczność dla bieżącego zbioru
            best_model_for_dataset = None  # Najlepszy model dla bieżącego zbioru

            # Przetwarzanie wszystkich wyników dla bieżącego zbioru
            for result in results:
                # Przygotuj informacje o modelu w czytelnym formacie
                model_info = {
                    "model": f"{result.get('algorithm', 'Unknown')} - v{result.get('model_version', 'Unknown')}",
                    "model_name": result.get("model_name", "Unknown"),
                    "effectiveness": result.get("effectiveness", 0),
                    "mae": result.get("MAE", 0),
                    "timestamp": result.get("timestamp", "Unknown")
                }
                dataset_results.append(model_info)

                # Sprawdź, czy to najlepszy model dla tego zbioru
                effectiveness = result.get("effectiveness", 0)
                if effectiveness > best_effectiveness:
                    best_effectiveness = effectiveness
                    best_model_for_dataset = {
                        "dataset": dataset,
                        "model": f"{result.get('algorithm', 'Unknown')} - v{result.get('model_version', 'Unknown')}",
                        "model_name": result.get("model_name", "Unknown"),
                        "effectiveness": effectiveness
                    }

            # Dodaj wyniki dla bieżącego zbioru do ogólnych wyników
            comparison_results.append({
                "dataset": dataset,
                "results": dataset_results,
                "best_model": best_model_for_dataset
            })

            # Sprawdź, czy to najlepszy model ogółem
            if best_effectiveness > overall_best_effectiveness:
                overall_best_effectiveness = best_effectiveness
                best_model_info = best_model_for_dataset

        # Zwróć wyniki porównania
        return {
            "results": comparison_results,
            "best_model": best_model_info
        }