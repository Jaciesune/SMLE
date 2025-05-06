import os
import shutil
import json
import logging
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from api.detection_api import DetectionAPI

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageDataset:
    def __init__(self, image_folder, annotation_path):
        self.image_folder = image_folder
        self.annotation_path = annotation_path
        logger.debug(f"[DEBUG] Inicjalizacja ImageDataset: image_folder={self.image_folder}, annotation_path={self.annotation_path}")
        
        # Wczytaj listę obrazów
        self.images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
        logger.debug(f"[DEBUG] Znalezione obrazy: {self.images}")
        if not self.images:
            logger.warning(f"[DEBUG] Brak obrazów w folderze {self.image_folder}")

        # Wczytaj annotacje (format LabelMe)
        self.file_to_annotations = {}
        for img_file in self.images:
            annotation_file = os.path.join(self.annotation_path, f"{os.path.splitext(img_file)[0]}.json")
            logger.debug(f"[DEBUG] Szukam annotacji dla {img_file}: {annotation_file}")
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    ann_data = json.load(f)
                self.file_to_annotations[img_file] = ann_data.get("shapes", [])
                logger.debug(f"[DEBUG] Znaleziono annotacje dla {img_file}: {len(self.file_to_annotations[img_file])} kształtów")
            else:
                logger.warning(f"[DEBUG] Brak pliku annotacji dla obrazu {img_file}")
                self.file_to_annotations[img_file] = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        img_path = os.path.join(self.image_folder, file_name)
        annotations = self.file_to_annotations.get(file_name, [])
        return img_path, annotations

class BenchmarkAPI:
    def __init__(self):
        self.detection_api = DetectionAPI()
        self.image_folder = "/app/backend/data/test/images"
        self.annotation_path = "/app/backend/data/test/annotations"
        self.history_file = "/app/backend/benchmark_history.json"
        logger.debug("[DEBUG] Inicjalizacja BenchmarkAPI")

        # Upewnij się, że foldery istnieją przy inicjalizacji
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotation_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        logger.debug(f"[DEBUG] Utworzono foldery przy inicjalizacji: images={self.image_folder}, annotations={self.annotation_path}")

    async def prepare_and_run_benchmark(self, images, annotations, request):
        # Krok 1: Przygotowanie danych (logika z prepare_benchmark_data)
        logger.debug(f"[DEBUG] Przygotowywanie danych: liczba obrazów={len(images)}, liczba annotacji={len(annotations)}")
        
        # Wyczyść stare dane
        if os.path.exists(self.image_folder):
            logger.debug(f"[DEBUG] Usuwanie starego folderu obrazów: {self.image_folder}")
            shutil.rmtree(self.image_folder)
        if os.path.exists(self.annotation_path):
            logger.debug(f"[DEBUG] Usuwanie starego folderu annotacji: {self.annotation_path}")
            shutil.rmtree(self.annotation_path)

        # Stwórz katalogi
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotation_path, exist_ok=True)
        logger.debug(f"[DEBUG] Utworzono nowe foldery: images={self.image_folder}, annotations={self.annotation_path}")

        # Zapisz obrazy
        for img_file in images:
            img_path = os.path.join(self.image_folder, img_file.filename)
            with open(img_path, "wb") as f:
                content = await img_file.read()
                if not content:
                    logger.error(f"[DEBUG] Obraz {img_file.filename} jest pusty!")
                    raise HTTPException(status_code=400, detail=f"Obraz {img_file.filename} jest pusty")
                f.write(content)
            logger.debug(f"[DEBUG] Zapisano obraz: {img_path}, rozmiar={os.path.getsize(img_path)} bajtów")

        # Zapisz annotacje
        for ann_file in annotations:
            ann_path = os.path.join(self.annotation_path, ann_file.filename)
            with open(ann_path, "wb") as f:
                content = await ann_file.read()
                if not content:
                    logger.error(f"[DEBUG] Annotacja {ann_file.filename} jest pusta!")
                    raise HTTPException(status_code=400, detail=f"Annotacja {ann_file.filename} jest pusta")
                f.write(content)
            logger.debug(f"[DEBUG] Zapisano annotację: {ann_path}, rozmiar={os.path.getsize(ann_path)} bajtów")

        # Sprawdź, czy foldery nie są puste
        images_list = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        annotations_list = [f for f in os.listdir(self.annotation_path) if f.endswith('.json')]
        logger.debug(f"[DEBUG] Po zapisie: obrazy={images_list}, annotacje={annotations_list}")
        if not images_list:
            logger.error("[DEBUG] Brak obrazów w folderze po zapisie!")
            raise HTTPException(status_code=500, detail="Brak obrazów w folderze po zapisie")
        if not annotations_list:
            logger.error("[DEBUG] Brak annotacji w folderze po zapisie!")
            raise HTTPException(status_code=500, detail="Brak annotacji w folderze po zapisie")

        # Krok 2: Uruchomienie benchmarku (logika z run_benchmark)
        algorithm = request.get("algorithm")
        model_version = request.get("model_version")
        model_name = request.get("model_name")
        source_folder = request.get("source_folder", "")

        logger.debug(f"[DEBUG] Uruchamianie benchmarku: algorithm={algorithm}, model_version={model_version}, model_name={model_name}, source_folder={source_folder}")

        # Sprawdź, czy foldery istnieją i nie są puste
        if not os.path.exists(self.image_folder):
            logger.error(f"[DEBUG] Folder z obrazami nie istnieje: {self.image_folder}")
            raise HTTPException(status_code=400, detail=f"Folder z obrazami nie istnieje: {self.image_folder}")
        if not os.path.exists(self.annotation_path):
            logger.error(f"[DEBUG] Folder z annotacjami nie istnieje: {self.annotation_path}")
            raise HTTPException(status_code=400, detail=f"Folder z annotacjami nie istnieje: {self.annotation_path}")

        images_list = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        annotations_list = [f for f in os.listdir(self.annotation_path) if f.endswith('.json')]
        logger.debug(f"[DEBUG] Przed benchmarkiem: obrazy={images_list}, annotacje={annotations_list}")
        if not images_list:
            logger.error(f"[DEBUG] Folder z obrazami jest pusty: {self.image_folder}")
            raise HTTPException(status_code=400, detail=f"Folder z obrazami jest pusty: {self.image_folder}")
        if not annotations_list:
            logger.error(f"[DEBUG] Folder z annotacjami jest pusty: {self.annotation_path}")
            raise HTTPException(status_code=400, detail=f"Folder z annotacjami jest pusty: {self.annotation_path}")

        # Wczytaj dane za pomocą ImageDataset
        dataset = ImageDataset(self.image_folder, self.annotation_path)
        if len(dataset) == 0:
            logger.error("[DEBUG] Brak obrazów w podanym folderze")
            raise HTTPException(status_code=400, detail="Brak obrazów w podanym folderze")

        # Mapowanie katalogów dla każdego algorytmu
        algorithm_to_path = {
            "Mask R-CNN": "/app/backend/Mask_RCNN/data/test/images",
            "MCNN": "/app/backend/MCNN/data/test/images",
            "FasterRCNN": "/app/backend/FasterRCNN/data/test/images",
        }
        if algorithm not in algorithm_to_path:
            logger.error(f"[DEBUG] Algorytm {algorithm} nie jest obsługiwany")
            raise HTTPException(status_code=400, detail=f"Algorytm {algorithm} nie jest obsługiwany")

        # Sprawdź, czy model istnieje
        model_paths = {
            "Mask R-CNN": "/app/backend/Mask_RCNN/models/",
            "MCNN": "/app/backend/MCNN/models/",
            "FasterRCNN": "/app/backend/FasterRCNN/saved_models/",
        }
        file_name = f"{model_name}_checkpoint.pth"
        model_path = os.path.join(model_paths.get(algorithm, ""), file_name)
        if not os.path.exists(model_path):
            logger.error(f"[DEBUG] Plik modelu nie istnieje: {model_path}")
            raise HTTPException(status_code=400, detail=f"Plik modelu nie istnieje: {model_path}")

        # Tworzenie katalogu na obrazy, jeśli nie istnieje
        container_images_path = algorithm_to_path[algorithm]
        os.makedirs(container_images_path, exist_ok=True)
        logger.debug(f"[DEBUG] Utworzono folder dla algorytmu: {container_images_path}")

        metrics_list = []
        ground_truth_counts = []
        predicted_counts = []
        for idx in range(len(dataset)):
            img_path, annotations = dataset[idx]
            logger.debug(f"[DEBUG] Przetwarzanie obrazu {idx+1}/{len(dataset)}: {img_path}")

            # Kopiowanie obrazu do katalogu algorytmu
            container_image_path = os.path.join(container_images_path, os.path.basename(img_path))
            logger.debug(f"[DEBUG] Kopiowanie obrazu z {img_path} do {container_image_path}")
            shutil.copy(img_path, container_image_path)

            try:
                # Uruchom detekcję
                logger.debug(f"[DEBUG] Uruchamianie detekcji na obrazie {os.path.basename(img_path)}")
                result, num_predicted = self.detection_api.analyze_with_model(container_image_path, algorithm, file_name)
                if "Błąd" in result:
                    logger.error(f"[DEBUG] Błąd detekcji dla {img_path}: {result}")
                    raise HTTPException(status_code=500, detail=f"Błąd detekcji dla {img_path}: {result}")
                logger.debug(f"[DEBUG] Liczba wykrytych rur: {num_predicted} dla {img_path}")

                # Policz obiekty z annotacji (format LabelMe)
                num_ground_truth = len(annotations)
                ground_truth_counts.append(num_ground_truth)
                predicted_counts.append(num_predicted)
                logger.debug(f"[DEBUG] Liczba rur w annotacji: {num_ground_truth} dla {img_path}")

                # Oblicz metrykę
                metric = abs(num_predicted - num_ground_truth)
                metrics_list.append(metric)
                logger.debug(f"[DEBUG] Metryka dla {img_path}: {metric}")
            finally:
                # Usuń skopiowany obraz
                if os.path.exists(container_image_path):
                    os.unlink(container_image_path)
                    logger.debug(f"[DEBUG] Usunięto skopiowany obraz: {container_image_path}")

        if metrics_list:
            # Oblicz MAE
            mae = sum(metrics_list) / len(metrics_list)
            # Oblicz średnią liczbę obiektów w ground truth
            avg_ground_truth = sum(ground_truth_counts) / len(ground_truth_counts) if ground_truth_counts else 1
            logger.debug(f"[DEBUG] Średnia liczba obiektów w ground truth: {avg_ground_truth}")
            # Oblicz średnią liczbę przewidywanych obiektów
            avg_predicted = sum(predicted_counts) / len(predicted_counts) if predicted_counts else 0
            logger.debug(f"[DEBUG] Średnia liczba przewidywanych obiektów: {avg_predicted}")
            # Oblicz skuteczność w procentach
            effectiveness = max(0, (1 - mae / avg_ground_truth)) * 100 if avg_ground_truth > 0 else 0
            results = {
                "MAE": mae,
                "effectiveness": round(effectiveness, 2),
                "algorithm": algorithm,
                "model_version": model_version,
                "model_name": model_name,
                "image_folder": self.image_folder,
                "source_folder": source_folder,
                "timestamp": datetime.now().isoformat()
            }
            logger.debug(f"[DEBUG] Wynik benchmarku: MAE={mae}, Skuteczność={effectiveness}%, avg_ground_truth={avg_ground_truth}, avg_predicted={avg_predicted}")

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
                history = []
                if os.path.exists(self.history_file):
                    try:
                        with open(self.history_file, "r") as f:
                            history = json.load(f)
                            if not isinstance(history, list):
                                history = []
                    except json.JSONDecodeError:
                        history = []

                found = False
                for i, entry in enumerate(history):
                    if (entry.get("model_name") == model_name and
                        entry.get("algorithm") == algorithm and
                        entry.get("model_version") == model_version and
                        entry.get("source_folder") == source_folder):
                        history[i] = results
                        found = True
                        logger.debug(f"[DEBUG] Nadpisz istniejący wpis w historii dla modelu {model_name}")
                        break

                if not found:
                    history.append(results)
                    logger.debug(f"[DEBUG] Dodano nowy wpis do historii dla modelu {model_name}")

                with open(self.history_file, "w") as f:
                    json.dump(history, f, indent=4)
                logger.debug("[DEBUG] Zaktualizowano historię benchmarków")
            except Exception as e:
                logger.error(f"[DEBUG] Błąd zapisu historii benchmarków: {e}")
                raise HTTPException(status_code=500, detail=f"Błąd zapisu historii: {e}")

            return results
        else:
            logger.error("[DEBUG] Brak przetworzonych par obraz-annotacja")
            raise HTTPException(status_code=400, detail="No valid image-annotation pairs processed")

    def get_benchmark_results(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    history = json.load(f)
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
        if not os.path.exists(self.history_file):
            logger.debug("[DEBUG] Brak historii benchmarków")
            return {"results": [], "best_model": None}

        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"[DEBUG] Błąd odczytu historii benchmarków: {e}")
            raise HTTPException(status_code=500, detail=f"Błąd odczytu historii: {e}")

        if not history:
            logger.debug("[DEBUG] Historia benchmarków jest pusta")
            return {"results": [], "best_model": None}

        results_by_dataset = {}
        for result in history:
            dataset = result.get("source_folder", "unknown_dataset")
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = []
            results_by_dataset[dataset].append(result)

        comparison_results = []
        best_model_info = None
        overall_best_effectiveness = -1

        for dataset, results in results_by_dataset.items():
            dataset_results = []
            best_effectiveness = -1
            best_model_for_dataset = None

            for result in results:
                model_info = {
                    "model": f"{result.get('algorithm', 'Unknown')} - v{result.get('model_version', 'Unknown')}",
                    "model_name": result.get("model_name", "Unknown"),
                    "effectiveness": result.get("effectiveness", 0),
                    "mae": result.get("MAE", 0),
                    "timestamp": result.get("timestamp", "Unknown")
                }
                dataset_results.append(model_info)

                effectiveness = result.get("effectiveness", 0)
                if effectiveness > best_effectiveness:
                    best_effectiveness = effectiveness
                    best_model_for_dataset = {
                        "dataset": dataset,
                        "model": f"{result.get('algorithm', 'Unknown')} - v{result.get('model_version', 'Unknown')}",
                        "model_name": result.get("model_name", "Unknown"),
                        "effectiveness": effectiveness
                    }

            comparison_results.append({
                "dataset": dataset,
                "results": dataset_results,
                "best_model": best_model_for_dataset
            })

            if best_effectiveness > overall_best_effectiveness:
                overall_best_effectiveness = best_effectiveness
                best_model_info = best_model_for_dataset

        return {
            "results": comparison_results,
            "best_model": best_model_info
        }