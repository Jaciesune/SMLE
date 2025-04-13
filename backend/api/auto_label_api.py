import os
from pathlib import Path
import shutil
import glob
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.Inne.Auto_Labeling.auto_label import main as auto_label_main

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AutoLabelAPI:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent  # backend
        self.algorithm = "Mask R-CNN"
        self.models_path = self.base_path / "Mask_RCNN" / "models"
        self.data_path = self.base_path / "data"
        logger.debug("Inicjalizacja AutoLabelAPI: base_path=%s, models_path=%s", self.base_path, self.models_path)

    def get_model_versions(self):
        """Zwraca listę plików modeli dla algorytmu Mask R-CNN."""
        if not self.models_path.exists():
            logger.warning("Katalog modeli %s nie istnieje.", self.models_path)
            return []
        model_versions = sorted([file.name for file in self.models_path.iterdir() if file.is_file() and file.name.endswith('_checkpoint.pth')])
        logger.info("Znalezione modele w %s: %s", self.models_path, model_versions)
        return model_versions

    def get_model_path(self, version):
        """Zwraca ścieżkę do wybranego modelu Mask R-CNN."""
        model_path = self.models_path / version
        if not model_path.exists() or not model_path.is_file():
            logger.error("Model %s nie istnieje.", model_path)
            return None
        if not model_path.name.endswith('_checkpoint.pth'):
            logger.error("Model %s nie kończy się na _checkpoint.pth.", model_path)
            return None
        logger.debug("Ścieżka modelu: %s", model_path)
        return str(model_path)  # Ścieżka w kontenerze

    def auto_label(self, input_dir, job_name, version, input_dir_docker, output_dir_docker, debug_dir_docker):
        """Przeprowadza automatyczne labelowanie katalogu zdjęć."""
        logger.debug("Rozpoczynam auto_label: job_name=%s, model_version=%s, input_dir_docker=%s",
                     job_name, version, input_dir_docker)
        model_path = self.get_model_path(version)
        if not model_path:
            error_msg = f"Błąd: Model {version} dla Mask R-CNN nie istnieje."
            logger.error(error_msg)
            return error_msg

        # Sprawdź input_dir_docker zamiast input_dir
        logger.debug("Sprawdzam katalog wejściowy w kontenerze: %s", input_dir_docker)
        if not os.path.exists(input_dir_docker):
            error_msg = f"Błąd: Katalog wejściowy w kontenerze {input_dir_docker} nie istnieje."
            logger.error(error_msg)
            return error_msg

        input_images = glob.glob(os.path.join(input_dir_docker, "*.jpg"))
        if not input_images:
            error_msg = f"Błąd: Brak obrazów .jpg w katalogu {input_dir_docker}."
            logger.error(error_msg)
            return error_msg

        logger.debug("Uruchamiam auto_label.py z argumentami: input_dir=%s, output_dir=%s, debug_dir=%s, model_path=%s",
                     input_dir_docker, output_dir_docker, debug_dir_docker, model_path)
        try:
            sys.argv = [
                "auto_label.py",
                "--input_dir", input_dir_docker,
                "--output_dir", output_dir_docker,
                "--debug_dir", debug_dir_docker,
                "--model_path", model_path
            ]
            auto_label_main()
            output_files = os.listdir(output_dir_docker) if os.path.exists(output_dir_docker) else []
            debug_files = os.listdir(debug_dir_docker) if os.path.exists(debug_dir_docker) else []
            logger.info("Zawartość katalogu wyjściowego %s: %s", output_dir_docker, output_files)
            logger.info("Zawartość katalogu debug %s: %s", debug_dir_docker, debug_files)
            if not output_files and not debug_files:
                error_msg = f"Błąd: Brak wyników w katalogach {output_dir_docker} ani {debug_dir_docker}."
                logger.error(error_msg)
                return error_msg
            return f"Labelowanie zakończone. Wyniki w {output_dir_docker}, debug w {debug_dir_docker}."
        except Exception as e:
            error_msg = f"Błąd podczas auto-labelingu: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg