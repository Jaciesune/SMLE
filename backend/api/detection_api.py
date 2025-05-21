"""
Moduł API detekcji obiektów (Detection API)

Ten moduł dostarcza interfejs do wykonywania detekcji obiektów na obrazach
przy użyciu różnych algorytmów i modeli (Mask R-CNN, FasterRCNN, MCNN).
Umożliwia wybór algorytmu, modelu oraz opcjonalne przetwarzanie wstępne obrazów.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na systemie plików
import sys               # Do modyfikacji ścieżek importu
from pathlib import Path  # Do wygodnego zarządzania ścieżkami
import subprocess        # Do uruchamiania skryptów zewnętrznych
import re                # Do parsowania wyników przy użyciu wyrażeń regularnych
import logging           # Do logowania informacji i błędów
import cv2               # Do operacji na obrazach

# Dodanie katalogu nadrzędnego do ścieżki importu, aby umożliwić import z modułów backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import funkcji przetwarzania wstępnego obrazów
from backend.Inne.preprocessing import preprocess_image 

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DetectionAPI:
    """
    Klasa API do detekcji obiektów w obrazach przy użyciu różnych algorytmów.
    
    Zapewnia jednolity interfejs do różnych modeli detekcji obiektów, w tym:
    - Mask R-CNN
    - FasterRCNN
    - MCNN
    """
    
    def __init__(self):
        """
        Inicjalizacja API detekcji z ustawieniem ścieżek do modeli dla różnych algorytmów.
        
        Ustawia bazową ścieżkę i mapuje nazwy algorytmów na ścieżki katalogów z modelami.
        """
        # Określ bazową ścieżkę jako katalog backend
        self.base_path = Path(__file__).resolve().parent.parent  # backend/
        
        # Definicja algorytmów i ich folderów z modelami
        self.algorithms = {
            "Mask R-CNN": self.base_path / "Mask_RCNN" / "models",
            "FasterRCNN": self.base_path / "FasterRCNN" / "saved_models",
            "MCNN": self.base_path / "MCNN" / "models"
        }
        logger.debug("Zainicjalizowano DetectionAPI z bazową ścieżką: %s", self.base_path)

    def get_algorithms(self):
        """
        Zwraca listę dostępnych algorytmów detekcji.
        
        Returns:
            list: Lista nazw dostępnych algorytmów (Mask R-CNN, FasterRCNN, MCNN)
        """
        logger.debug("Pobieranie listy algorytmów: %s", list(self.algorithms.keys()))
        return list(self.algorithms.keys())

    def get_model_versions(self, algorithm):
        """
        Zwraca listę plików modeli dla wybranego algorytmu.
        
        Dla Mask R-CNN zwracane są tylko pliki z rozszerzeniem *_checkpoint.pth,
        dla pozostałych algorytmów zwracane są wszystkie pliki w katalogu modeli.
        
        Parameters:
            algorithm (str): Nazwa algorytmu detekcji
            
        Returns:
            list: Lista nazw plików modeli dla wybranego algorytmu, posortowana alfabetycznie.
                  Pusta lista, jeśli algorytm nie istnieje lub katalog modeli nie istnieje.
        """
        # Sprawdź, czy algorytm jest obsługiwany
        if algorithm not in self.algorithms:
            logger.warning("Algorytm %s nie jest obsługiwany", algorithm)
            return []

        # Pobierz ścieżkę do katalogu z modelami
        model_path = self.algorithms[algorithm]
        if not model_path.exists():
            logger.warning("Katalog modeli %s nie istnieje", model_path)
            return []

        # Zwróć odpowiednie pliki w zależności od algorytmu
        if algorithm == "Mask R-CNN":
            # Dla Mask R-CNN zwracamy tylko pliki z końcówką *_checkpoint.pth
            models = sorted([file.name for file in model_path.iterdir() 
                         if file.is_file() and file.name.endswith('_checkpoint.pth')])
        else:
            # Dla pozostałych algorytmów zwracamy wszystkie pliki w katalogu
            models = sorted([file.name for file in model_path.iterdir() if file.is_file()])
        
        logger.debug("Znalezione modele dla algorytmu %s: %s", algorithm, models)
        return models

    def get_model_path(self, algorithm, version):
        """
        Zwraca pełną ścieżkę do wybranego modelu.
        
        Dla Mask R-CNN akceptowane są tylko pliki z końcówką *_checkpoint.pth,
        dla pozostałych algorytmów dowolne pliki.
        
        Parameters:
            algorithm (str): Nazwa algorytmu detekcji
            version (str): Nazwa pliku modelu
            
        Returns:
            str: Pełna ścieżka do pliku modelu jako string, lub None jeśli model nie istnieje
                 lub ma nieprawidłowy format
        """
        # Sprawdź, czy algorytm jest obsługiwany
        if algorithm not in self.algorithms:
            logger.warning("Algorytm %s nie jest obsługiwany", algorithm)
            return None

        # Utwórz pełną ścieżkę do modelu
        model_path = self.algorithms[algorithm] / version
        if not model_path.exists() or not model_path.is_file():
            logger.warning("Model %s nie istnieje", model_path)
            return None

        # Sprawdź format pliku w zależności od algorytmu
        if algorithm == "Mask R-CNN":
            if model_path.name.endswith('_checkpoint.pth'):
                logger.debug("Znaleziono poprawny model Mask R-CNN: %s", model_path)
                return str(model_path)
            logger.warning("Model Mask R-CNN musi kończyć się na _checkpoint.pth")
            return None
        else:
            logger.debug("Znaleziono model %s: %s", algorithm, model_path)
            return str(model_path)
        
    def run_script(self, script_name, algorithm, *args):
        """
        Uruchamia skrypt bezpośrednio w bieżącym środowisku (kontenerze backend-app).
        
        Wybiera odpowiednią ścieżkę do skryptu na podstawie wybranego algorytmu,
        a następnie wykonuje go z podanymi argumentami.
        
        Parameters:
            script_name (str): Nazwa skryptu do uruchomienia
            algorithm (str): Nazwa algorytmu (określa lokalizację skryptu)
            *args: Argumenty przekazywane do skryptu
            
        Returns:
            str: Wynik działania skryptu (stdout) lub komunikat o błędzie
        """
        try:
            # Mapowanie ścieżek dla każdego algorytmu
            if algorithm == "Mask R-CNN":
                script_path = f"/app/backend/Mask_RCNN/scripts/{script_name}"
            elif algorithm == "MCNN":
                script_path = f"/app/backend/MCNN/{script_name}"
            elif algorithm == "FasterRCNN":
                script_path = f"/app/backend/FasterRCNN/{script_name}"
            else:
                logger.error("Algorytm %s nie jest obsługiwany", algorithm)
                return f"Błąd: Algorytm {algorithm} nie jest obsługiwany."

            # Budowanie polecenia do uruchomienia skryptu
            command = ["python", script_path, *args]

            logger.debug("Uruchamiam polecenie: %s", ' '.join(command))
            # Uruchomienie skryptu i przechwycenie jego wyjścia
            result = subprocess.run(
                command,
                capture_output=True,  # Przechwytuje stdout i stderr
                text=True,            # Konwertuje wynik na tekst
                encoding="utf-8",     # Określa kodowanie
                errors="replace"      # Zastępuje znaki, których nie można zdekodować
            )
            
            # Sprawdź, czy skrypt zakończył się pomyślnie
            if result.returncode != 0:
                logger.error("Błąd w skrypcie %s: stderr=%s", script_name, result.stderr)
                return f"Błąd podczas uruchamiania skryptu {script_name}: {result.stderr}"
            
            logger.debug("Wynik skryptu %s: stdout=%s", script_name, result.stdout)
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            # Obsługa błędów wywoływania procesu
            logger.error("Błąd podczas uruchamiania skryptu: %s", str(e))
            return f"Błąd podczas uruchamiania skryptu: {e}"
        except Exception as e:
            # Obsługa innych wyjątków
            logger.error("Nieoczekiwany błąd: %s", str(e))
            return f"Nieoczekiwany błąd: {e}"
        
    def analyze_with_model(self, image_path, algorithm, version, preprocessing=False):
        """
        Przeprowadza detekcję na obrazie przy użyciu wybranego modelu.
        
        Wykonuje opcjonalne przetwarzanie wstępne obrazu, a następnie
        uruchamia odpowiedni skrypt detekcji dla wybranego algorytmu i modelu.
        
        Parameters:
            image_path (str): Ścieżka do pliku obrazu
            algorithm (str): Nazwa algorytmu detekcji
            version (str): Nazwa pliku modelu
            preprocessing (bool, optional): Czy przeprowadzić preprocessing obrazu. Domyślnie False.
            
        Returns:
            tuple: Para (wynik, liczba_detekcji), gdzie:
                - wynik (str): Ścieżka do wynikowego obrazu lub komunikat o błędzie
                - liczba_detekcji (int): Liczba wykrytych obiektów
        """
        # Sprawdź, czy model istnieje
        model_path = self.get_model_path(algorithm, version)
        if not model_path:
            logger.error("Model %s dla algorytmu %s nie istnieje", version, algorithm)
            return f"Błąd: Model {version} dla {algorithm} nie istnieje.", 0

        # Sprawdź, czy obraz istnieje
        if not os.path.exists(image_path):
            logger.error("Obraz %s nie istnieje", image_path)
            return f"Błąd: Obraz {image_path} nie istnieje.", 0

        #######################
        # Konfiguracja ścieżek dla algorytmów
        #######################
        # Mapowanie ścieżek dla każdego algorytmu
        if algorithm == "Mask R-CNN":
            host_detectes_path = self.base_path / "Mask_RCNN" / "data" / "detectes"
            container_base_path = "/app/backend/Mask_RCNN"
            script_name = "detect.py"
        elif algorithm == "MCNN":
            host_detectes_path = self.base_path / "MCNN" / "data" / "detectes"
            container_base_path = "/app/backend/MCNN"
            script_name = "test_model.py"
        elif algorithm == "FasterRCNN":
            host_detectes_path = self.base_path / "FasterRCNN" / "data" / "detectes"
            container_base_path = "/app/backend/FasterRCNN"
            script_name = "test.py"
        else:
            logger.error("Algorytm %s nie jest obsługiwany", algorithm)
            return f"Błąd: Algorytm {algorithm} nie jest obsługiwany.", 0

        # Utwórz katalog na wyniki detekcji, jeśli nie istnieje
        host_detectes_path.mkdir(parents=True, exist_ok=True)

        #######################
        # Preprocessing obrazu (opcjonalny)
        #######################
        # Jeśli włączony preprocessing
        if preprocessing:
            image_name = os.path.basename(image_path)
            # Ustal ścieżkę do przetworzonego obrazu w podkatalogu preprocessed
            preprocessed_image_name = image_name  # Nazwa pozostaje taka sama
            preprocessed_image_dir = os.path.join(os.path.dirname(image_path), "preprocessed")
            preprocessed_image_path = os.path.join(preprocessed_image_dir, preprocessed_image_name)
            container_preprocessed_image_path = f"{container_base_path}/data/test/images/preprocessed/{image_name}"

            try:
                logger.debug("Uruchamiam preprocessing dla obrazu: %s", image_path)
                # Bezpośrednie wywołanie preprocess_image
                os.makedirs(preprocessed_image_dir, exist_ok=True)
                processed_image = preprocess_image(image_path, preprocessed_image_path)
                if processed_image is None:
                    logger.error("Błąd w preprocess_image: Nie udało się przetworzyć obrazu")
                    return f"Błąd podczas preprocessingu: Nie udało się przetworzyć obrazu", 0
                # Zapisz przetworzony obraz
                cv2.imwrite(preprocessed_image_path, processed_image)
                logger.debug("Preprocessing zakończony, zapisano: %s", preprocessed_image_path)
                # Aktualizuj image_path na przetworzony obraz
                image_path = preprocessed_image_path
            except Exception as e:
                logger.error("Wyjątek podczas preprocessingu: %s", str(e))
                return f"Błąd podczas preprocessingu: {e}", 0
        else:
            container_preprocessed_image_path = f"{container_base_path}/data/test/images/{os.path.basename(image_path)}"

        #######################
        # Detekcja obiektów
        #######################
        # Ścieżki w kontenerze
        image_name = os.path.basename(image_path)
        container_image_path = container_preprocessed_image_path
        container_model_path = f"{container_base_path}/{'models' if algorithm != 'FasterRCNN' else 'saved_models'}/{version}"

        # Uruchomienie detekcji
        if algorithm == "FasterRCNN":
            # FasterRCNN wymaga innych parametrów niż pozostałe algorytmy
            result = self.run_script(
                script_name,
                algorithm,
                "--image_path", container_image_path,
                "--model_path", container_model_path,
                "--output_dir", f"{container_base_path}/data/detectes",
                "--threshold", "0.25",  # Próg pewności detekcji
                "--num_classes", "2"    # Liczba klas (tło + obiekt)
            )
        else:
            # Pozostałe algorytmy przyjmują prostsze parametry
            result = self.run_script(script_name, algorithm, container_image_path, container_model_path)

        # Sprawdź, czy wystąpił błąd podczas detekcji
        if "Błąd" in result:
            logger.error("Błąd w wyniku detekcji: %s", result)
            return result, 0

        #######################
        # Przetwarzanie wyników
        #######################
        # Sprawdź, czy plik wynikowy został utworzony
        result_image_name = os.path.splitext(image_name)[0] + "_detected.jpg"
        result_path = host_detectes_path / result_image_name
        if not result_path.exists():
            logger.error("Wynik detekcji nie został zapisany w %s", result_path)
            return f"Błąd: Wynik detekcji nie został zapisany w {result_path}.", 0

        # Wyciągnij liczbę detekcji z wyniku skryptu
        detections_count = 0
        match = re.search(r"Detections: (\d+)", result)
        if match:
            detections_count = int(match.group(1))
            logger.debug("Znaleziono %d detekcji", detections_count)

        return str(result_path), detections_count