"""
Moduł API automatycznego oznaczania obrazów (Auto Label API)

Ten moduł dostarcza interfejs do automatycznego oznaczania obrazów przy użyciu
modeli Mask R-CNN. Umożliwia wykrywanie obiektów na obrazach i generowanie
plików adnotacji w formacie LabelMe, które mogą być później używane do treningu
lub wizualizacji.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na systemie plików
from pathlib import Path  # Do wygodnego zarządzania ścieżkami
import shutil            # Do kopiowania i usuwania plików
import glob              # Do wyszukiwania plików według wzorca
import sys               # Do modyfikacji ścieżek importu i argumentów
import logging           # Do logowania informacji i błędów

# Dodanie katalogu nadrzędnego do ścieżki importu, aby umożliwić import z modułów backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import funkcji głównej z modułu auto_label
from backend.Inne.Auto_Labeling.auto_label import main as auto_label_main

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AutoLabelAPI:
    """
    Klasa API do automatycznego oznaczania obrazów przy użyciu modeli Mask R-CNN.
    
    Zapewnia interfejs do wykrywania obiektów na obrazach i generowania
    odpowiednich plików adnotacji w formacie LabelMe.
    """
    
    def __init__(self):
        """
        Inicjalizacja API auto-labelingu z ustawieniem ścieżek do modeli i danych.
        
        Konfiguruje ścieżki bazowe, katalog modeli oraz inne wymagane katalogi.
        """
        # Ścieżka bazowa (katalog backend)
        self.base_path = Path(__file__).resolve().parent.parent
        
        # Zawsze używamy algorytmu Mask R-CNN do auto-labelingu
        self.algorithm = "Mask R-CNN"
        
        # Ścieżka do katalogu z modelami Mask R-CNN
        self.models_path = self.base_path / "Mask_RCNN" / "models"
        
        # Ścieżka do katalogu z danymi
        self.data_path = self.base_path / "data"
        
        logger.debug("Inicjalizacja AutoLabelAPI: base_path=%s, models_path=%s", self.base_path, self.models_path)

    def get_model_versions(self):
        """
        Zwraca listę dostępnych wersji modeli Mask R-CNN.
        
        Returns:
            list: Lista nazw plików modeli (tylko pliki z końcówką *_checkpoint.pth), posortowana alfabetycznie.
                  Pusta lista, jeśli katalog modeli nie istnieje.
        """
        # Sprawdź, czy katalog modeli istnieje
        if not self.models_path.exists():
            logger.warning("Katalog modeli %s nie istnieje.", self.models_path)
            return []
        
        # Znajdź wszystkie pliki modeli i posortuj je
        model_versions = sorted([file.name for file in self.models_path.iterdir() 
                              if file.is_file() and file.name.endswith('_checkpoint.pth')])
        
        logger.info("Znalezione modele w %s: %s", self.models_path, model_versions)
        return model_versions

    def get_model_path(self, version):
        """
        Zwraca pełną ścieżkę do wybranego modelu po weryfikacji jego poprawności.
        
        Parameters:
            version (str): Nazwa pliku modelu
            
        Returns:
            str: Pełna ścieżka do pliku modelu jako string, lub None jeśli model nie istnieje
                 lub ma nieprawidłowy format (musi kończyć się na _checkpoint.pth)
        """
        # Utwórz pełną ścieżkę do modelu
        model_path = self.models_path / version
        
        # Weryfikacja: czy plik istnieje
        if not model_path.exists() or not model_path.is_file():
            logger.error("Model %s nie istnieje.", model_path)
            return None
        
        # Weryfikacja: czy ma poprawną końcówkę nazwy
        if not model_path.name.endswith('_checkpoint.pth'):
            logger.error("Model %s nie kończy się na _checkpoint.pth.", model_path)
            return None
            
        logger.debug("Ścieżka modelu: %s", model_path)
        return str(model_path)

    def auto_label(self, input_dir, job_name, version, input_dir_docker, output_dir_docker, debug_dir_docker, custom_label):
        """
        Przeprowadza automatyczne oznaczanie obrazów przy użyciu wybranego modelu Mask R-CNN.
        
        Wykorzystuje skrypt auto_label.py do wykrywania obiektów na obrazach
        i generowania plików adnotacji w formacie LabelMe.
        
        Parameters:
            input_dir (str): Ścieżka do katalogu wejściowego z obrazami
            job_name (str): Nazwa zadania (do identyfikacji)
            version (str): Nazwa pliku modelu Mask R-CNN
            input_dir_docker (str): Ścieżka do katalogu wejściowego w kontenerze Docker
            output_dir_docker (str): Ścieżka do katalogu wyjściowego w kontenerze Docker
            debug_dir_docker (str, optional): Ścieżka do katalogu debugowania w kontenerze Docker, lub None
            custom_label (str): Niestandardowa etykieta do przypisania wykrytym obiektom
            
        Returns:
            str: Komunikat o wyniku operacji (sukces lub opis błędu)
        """
        #######################
        # Walidacja parametrów
        #######################
        logger.debug("Rozpoczynam auto_label: job_name=%s, model_version=%s, input_dir_docker=%s, custom_label=%s",
                     job_name, version, input_dir_docker, custom_label)
        
        # Sprawdź, czy model istnieje i ma poprawny format
        model_path = self.get_model_path(version)
        if not model_path:
            error_msg = f"Błąd: Model {version} dla Mask R-CNN nie istnieje."
            logger.error(error_msg)
            return error_msg

        # Sprawdź, czy katalog wejściowy istnieje
        logger.debug("Sprawdzam katalog wejściowy w kontenerze: %s", input_dir_docker)
        if not os.path.exists(input_dir_docker):
            error_msg = f"Błąd: Katalog wejściowy w kontenerze {input_dir_docker} nie istnieje."
            logger.error(error_msg)
            return error_msg

        # Sprawdź, czy są obrazy do przetworzenia
        input_images = glob.glob(os.path.join(input_dir_docker, "*.jpg"))
        logger.debug(f"Znalezione obrazy w {input_dir_docker}: {input_images}")
        if not input_images:
            error_msg = f"Błąd: Brak obrazów .jpg w katalogu {input_dir_docker}."
            logger.error(error_msg)
            return error_msg

        #######################
        # Uruchomienie auto-labelingu
        #######################
        logger.debug("Uruchamiam auto_label.py z argumentami: input_dir=%s, output_dir=%s, debug_dir=%s, model_path=%s, custom_label=%s",
                     input_dir_docker, output_dir_docker, debug_dir_docker, model_path, custom_label)
        
        try:
            # Przygotowanie argumentów dla auto_label_main
            sys.argv = [
                "auto_label.py",  # Nazwa skryptu (nieistotna, ale wymagana)
                "--input_dir", input_dir_docker,  # Katalog wejściowy
                "--output_dir", output_dir_docker,  # Katalog wyjściowy
                "--debug_dir", debug_dir_docker if debug_dir_docker else "",  # Katalog debugowania (opcjonalny)
                "--model_path", model_path,  # Ścieżka do modelu
                "--custom_label", custom_label  # Niestandardowa etykieta
            ]
            
            logger.debug("Argumenty przekazane do auto_label_main: %s", sys.argv)
            
            # Wywołanie funkcji main z modułu auto_label
            auto_label_main()
            
            #######################
            # Weryfikacja wyników
            #######################
            output_files = os.listdir(output_dir_docker) if os.path.exists(output_dir_docker) else []
            logger.info("Zawartość katalogu wyjściowego %s: %s", output_dir_docker, output_files)
            
            # Sprawdź, czy zostały wygenerowane jakieś pliki
            if not output_files:
                error_msg = f"Błąd: Brak wyników w katalogu {output_dir_docker}. Możliwe, że model nie wykrył żadnych obiektów."
                logger.warning(error_msg)
                return error_msg
                
            return f"Labelowanie zakończone. Wyniki w {output_dir_docker}."
            
        except Exception as e:
            # Obsługa błędów podczas auto-labelingu
            error_msg = f"Błąd podczas auto-labelingu: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg