"""
Moduł API zarządzania zbiorami danych (Dataset API)

Ten moduł dostarcza interfejs do tworzenia, zarządzania i manipulowania zbiorami danych
dla modeli uczenia maszynowego. Umożliwia konwersję formatów adnotacji (LabelMe do COCO),
podział danych na podzbiory treningowe/walidacyjne/testowe oraz pakowanie zbiorów danych.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na ścieżkach i plikach
import json              # Do operacji na plikach JSON
import base64            # Do dekodowania masek w formacie Base64
import shutil            # Do kopiowania i usuwania plików/katalogów
import zipfile           # Do tworzenia archiwów ZIP z danymi
import numpy as np       # Do operacji na tablicach numerycznych (dla masek)
from PIL import Image    # Do operacji na obrazach
from pycocotools import mask as mask_utils  # Do konwersji masek na format COCO RLE
from io import BytesIO   # Do obsługi strumieni bajtów (dla masek)
import logging           # Do logowania informacji i błędów
import random            # Do losowego podziału danych na podzbiory

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAPI:
    """
    Klasa API do zarządzania zbiorami danych uczenia maszynowego.
    
    Dostarcza metody do tworzenia, modyfikacji, eksportu i usuwania zbiorów danych,
    a także konwersji między różnymi formatami adnotacji (np. LabelMe do COCO).
    """
    
    def __init__(self):
        """
        Inicjalizacja API z domyślnymi ścieżkami i ustawieniami.
        
        Ustawia bazowy katalog wyjściowy dla zbiorów danych oraz listę dozwolonych
        rozszerzeń plików obrazów.
        """
        # Katalog bazowy dla wszystkich zbiorów danych
        self.output_base_dir = "/app/backend/data/dataset_create"
        
        # Dozwolone rozszerzenia plików obrazów
        self.allowed_image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        # Upewnij się, że katalog bazowy istnieje
        os.makedirs(self.output_base_dir, exist_ok=True)
        logger.info(f"Zainicjalizowano DatasetAPI z katalogiem bazowym: {self.output_base_dir}")

    #######################
    # Funkcje pomocnicze
    #######################
    
    def decode_mask(self, base64_str):
        """
        Dekoduje maskę zapisaną w formacie Base64 do postaci macierzy NumPy.
        
        Parameters:
            base64_str (str): Ciąg znaków Base64 zawierający zakodowaną maskę
            
        Returns:
            numpy.ndarray: Maska w postaci macierzy binarnej (0/1)
            
        Raises:
            ValueError: Gdy wystąpi błąd podczas dekodowania maski
        """
        try:
            # Dekoduj ciąg Base64 do bajtów
            mask_bytes = base64.b64decode(base64_str)
            
            # Konwertuj bajty na obraz i przekształć na obraz w skali szarości
            mask_image = Image.open(BytesIO(mask_bytes)).convert("L")
            
            # Konwertuj obraz na tablicę NumPy
            return np.array(mask_image, dtype=np.uint8)
        except Exception as e:
            logger.error("Błąd dekodowania maski: %s", e)
            raise ValueError(f"Błąd dekodowania maski: {e}")

    def normalize_image_path(self, image_path):
        """
        Normalizuje ścieżkę obrazu do samej nazwy pliku.
        
        Parameters:
            image_path (str): Ścieżka do pliku obrazu
            
        Returns:
            str: Sama nazwa pliku bez ścieżki katalogu
        """
        # Wyodrębnij samą nazwę pliku, bez ścieżki katalogu
        return os.path.basename(image_path)

    def load_labelme_json(self, json_path):
        """
        Wczytuje plik JSON z LabelMe i normalizuje ścieżkę obrazu.
        
        Parameters:
            json_path (str): Ścieżka do pliku JSON w formacie LabelMe
            
        Returns:
            dict: Dane adnotacji z pliku LabelMe z znormalizowaną ścieżką obrazu
        """
        # Wczytaj plik JSON
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Znormalizuj ścieżkę obrazu do samej nazwy pliku
        data["imagePath"] = self.normalize_image_path(data["imagePath"])
        return data

    def encode_rle(self, mask):
        """
        Koduje maskę binarną do formatu Run-Length Encoding (RLE) używanego przez COCO.
        
        Parameters:
            mask (numpy.ndarray): Binarna maska w formacie macierzy NumPy
            
        Returns:
            dict: Maska zakodowana w formacie RLE
        """
        # Przekształć tablicę do formatu column-major (dla biblioteki pycocotools)
        mask = np.asfortranarray(mask)
        
        # Koduj maskę w formacie RLE
        return mask_utils.encode(mask)

    #######################
    # Konwersja formatów adnotacji
    #######################
    
    def convert_to_coco(self, data_list, output_path):
        """
        Konwertuje listę adnotacji LabelMe do formatu COCO JSON z maskami w formacie RLE.
        
        Parameters:
            data_list (list): Lista adnotacji w formacie LabelMe
            output_path (str): Ścieżka do pliku wyjściowego JSON w formacie COCO
            
        Returns:
            None: Wynik jest zapisywany do pliku określonego w output_path
        """
        # Inicjalizacja struktury COCO JSON
        coco = {
            "info": {"description": "Dataset converted from LabelMe"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        category_map = {}  # Mapowanie nazw kategorii na ID
        annotation_id = 1  # Licznik ID adnotacji

        # Przetwarzanie każdego obrazu
        for image_id, data in enumerate(data_list):
            # Dodaj informacje o obrazie
            coco["images"].append({
                "id": image_id,
                "file_name": data["imagePath"],
                "width": data["imageWidth"],
                "height": data["imageHeight"]
            })

            # Przetwarzanie każdego kształtu (obiektu) na obrazie
            for shape in data["shapes"]:
                label = shape["label"]
                
                # Dodaj nową kategorię, jeśli nie istnieje
                if label not in category_map:
                    category_map[label] = len(category_map) + 1
                    coco["categories"].append({"id": category_map[label], "name": label})

                # Oblicz bounding box
                try:
                    xmin, ymin = map(int, shape["points"][0])
                    xmax, ymax = map(int, shape["points"][1])
                    width, height = xmax - xmin, ymax - ymin
                except Exception as e:
                    logger.error("Błąd w bbox dla %s: %s", data["imagePath"], e)
                    continue

                # Przygotuj podstawową adnotację
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[label],
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0
                }

                # Dodaj segmentację, jeśli dostępna maska
                if "mask" in shape:
                    try:
                        # Dekoduj maskę z base64
                        mask_array = self.decode_mask(shape["mask"])
                        
                        # Koduj maskę w formacie RLE
                        rle = self.encode_rle(mask_array)
                        
                        # Zapisz zakodowaną maskę
                        annotation["segmentation"] = {
                            "counts": rle["counts"].decode("utf-8"),
                            "size": list(rle["size"])
                        }
                    except ValueError as e:
                        logger.warning("Pomijam maskę dla %s: %s", data["imagePath"], e)
                        continue

                # Dodaj adnotację do listy
                coco["annotations"].append(annotation)
                annotation_id += 1

        # Zapisz wynikowy plik COCO JSON
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=4)
        
        logger.info(f"Skonwertowano {len(data_list)} obrazów do formatu COCO w pliku {output_path}")

    #######################
    # Zarządzanie zbiorami danych
    #######################

    def check_duplicates(self, username, dataset_name, image_name):
        """
        Sprawdza, czy obraz o danej nazwie już istnieje w którejś z kategorii zbioru danych.
        
        Parameters:
            username (str): Nazwa użytkownika
            dataset_name (str): Nazwa zbioru danych
            image_name (str): Nazwa pliku obrazu do sprawdzenia
            
        Returns:
            bool: True, jeśli obraz już istnieje w zbiorze danych, False w przeciwnym przypadku
        """
        # Katalog użytkownika zawierający zbiór danych
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        
        # Sprawdź każdy podzbiór
        for subset in ["train", "val", "test"]:
            subset_dir = os.path.join(user_dir, subset)
            
            # Ścieżka do obrazów zależy od rodzaju podzbioru
            if subset in ["train", "val"]:
                subset_dir = os.path.join(subset_dir, "images")
            
            # Sprawdź, czy obraz istnieje w katalogu
            if os.path.exists(subset_dir) and image_name in os.listdir(subset_dir):
                return True
        
        return False

    def manual_split(self, files, train_ratio, val_ratio, test_ratio):
        """
        Ręcznie dzieli listę plików na zbiory treningowy, walidacyjny i testowy według podanych proporcji.
        
        Podziału dokonuje w sposób losowy, zachowując proporcje określone w parametrach.
        
        Parameters:
            files (list): Lista nazw plików do podziału
            train_ratio (float): Proporcja zbioru treningowego (0-1)
            val_ratio (float): Proporcja zbioru walidacyjnego (0-1)
            test_ratio (float): Proporcja zbioru testowego (0-1)
            
        Returns:
            tuple: Krotka trzech list (train_files, val_files, test_files) zawierających
                  nazwy plików przypisane do każdego zbioru
        """
        # Sprawdź, czy lista plików nie jest pusta
        total = len(files)
        if total == 0:
            return [], [], []

        # Losowo pomieszaj pliki
        random.shuffle(files)
        
        # Oblicz granice podziału
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        test_end = total

        # Podziel listę plików na trzy części
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:test_end]

        logger.info(f"Podzielono {total} plików: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        return train_files, val_files, test_files

    def create_dataset(self, username, dataset_name, input_files, train_ratio, val_ratio, test_ratio):
        """
        Tworzy zbiór danych z przesłanych plików, dzieląc je na podzbiory treningowy, walidacyjny i testowy.
        
        Parameters:
            username (str): Nazwa użytkownika
            dataset_name (str): Nazwa zbioru danych
            input_files (list): Lista obiektów plików (UploadFile) do przetworzenia
            train_ratio (float): Proporcja zbioru treningowego (0-1)
            val_ratio (float): Proporcja zbioru walidacyjnego (0-1)
            test_ratio (float): Proporcja zbioru testowego (0-1)
            
        Returns:
            bool: True, jeśli tworzenie zbioru danych powiodło się
            
        Raises:
            ValueError: Gdy brakuje plików JSON lub obrazów, lub gdy inne błędy uniemożliwiają utworzenie zbioru
        """
        # Tworzenie katalogów do przechowywania plików
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        input_dir = os.path.join(user_dir, "input")  # Tymczasowy katalog na przesłane pliki
        output_dir = os.path.join(user_dir, "output")  # Tymczasowy katalog na wyniki

        try:
            # Utwórz katalogi dla przesłanych plików
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Utworzono katalogi dla datasetu {dataset_name}: {input_dir}, {output_dir}")

            #######################
            # Zapisywanie przesłanych plików
            #######################
            
            json_files = []  # Lista plików JSON
            image_files = []  # Lista plików obrazów
            
            # Przetwarzanie każdego przesłanego pliku
            for file in input_files:
                # Ścieżka docelowa dla pliku
                file_path = os.path.join(input_dir, file.filename)
                
                # Sprawdź rozszerzenie pliku
                _, ext = os.path.splitext(file.filename.lower())
                
                # Zapisz plik obrazu
                if ext in self.allowed_image_extensions:
                    with open(file_path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                    image_files.append(file.filename)
                    logger.debug(f"Zapisano obraz: {file.filename}")
                
                # Zapisz plik JSON
                elif ext == ".json":
                    with open(file_path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                    json_files.append(file.filename)
                    logger.debug(f"Zapisano JSON: {file.filename}")
                
                # Pomiń pliki o nieobsługiwanych rozszerzeniach
                else:
                    logger.warning(f"Pomijam plik o nieobsługiwanym rozszerzeniu: {file.filename}")

            # Sprawdź, czy przesłano pliki JSON
            if not json_files:
                raise ValueError("Brak plików .json w przesłanych danych.")
                
            # Sprawdź, czy przesłano pliki obrazów
            if not image_files:
                raise ValueError("Brak plików obrazów (.jpg, .jpeg, .png, .bmp) w przesłanych danych.")

            #######################
            # Parowanie plików i sprawdzanie duplikatów
            #######################
            
            # Lista obrazów, które mają odpowiadające pliki JSON i nie są duplikatami
            paired_files = []
            
            # Przetwarzanie każdego obrazu
            for img in image_files:
                # Nazwa pliku JSON odpowiadającego obrazowi
                json_name = img.rsplit(".", 1)[0] + ".json"
                
                # Sprawdź, czy istnieje odpowiadający plik JSON
                if json_name not in json_files:
                    logger.warning(f"Brak pliku JSON dla obrazu {img}, pomijam.")
                    continue
                
                # Sprawdź, czy obraz nie jest już w zbiorze danych
                if self.check_duplicates(username, dataset_name, img):
                    logger.warning(f"Obraz {img} już istnieje w datasecie, pomijam.")
                    continue
                
                # Dodaj do listy obrazów do przetworzenia
                paired_files.append(img)

            # Sprawdź, czy zostały jakieś pliki do przetworzenia
            if not paired_files:
                raise ValueError("Brak nowych par obraz-JSON do przetworzenia.")

            #######################
            # Podział na podzbiory i przetwarzanie
            #######################
            
            # Podziel pliki na zbiory treningowy, walidacyjny i testowy
            train_files, val_files, test_files = self.manual_split(
                paired_files, train_ratio, val_ratio, test_ratio
            )

            # Słownik przypisujący pliki do odpowiednich podzbiorów
            datasets = {
                "train": train_files,
                "val": val_files,
                "test": test_files
            }

            # Przetwarzanie każdego podzbioru
            for dataset, files in datasets.items():
                # Katalog dla podzbioru
                dataset_dir = os.path.join(user_dir, dataset)
                
                # Tworzenie katalogów dla obrazów i adnotacji (dla train i val)
                if dataset != "test":
                    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
                    os.makedirs(os.path.join(dataset_dir, "annotations"), exist_ok=True)
                else:
                    os.makedirs(dataset_dir, exist_ok=True)

                # Lista danych do konwersji na format COCO
                data_list = []
                
                # Przetwarzanie każdego obrazu w podzbiorze
                for img_file in files:
                    # Nazwa pliku JSON odpowiadającego obrazowi
                    json_file = img_file.rsplit(".", 1)[0] + ".json"
                    json_path = os.path.join(input_dir, json_file)
                    
                    try:
                        # Wczytaj dane adnotacji LabelMe
                        labelme_data = self.load_labelme_json(json_path)
                        image_name = labelme_data["imagePath"]
                        
                        # Sprawdź, czy plik to obraz o obsługiwanym rozszerzeniu
                        _, ext = os.path.splitext(image_name.lower())
                        if ext not in self.allowed_image_extensions:
                            logger.warning(f"Plik {image_name} nie jest obrazem (rozszerzenie {ext}), pomijam.")
                            continue

                        # Sprawdź, czy obraz istnieje
                        source_image_path = os.path.join(input_dir, image_name)
                        if not os.path.exists(source_image_path):
                            logger.warning(f"Brak obrazu {source_image_path}, pomijam.")
                            continue

                        # Kopiowanie plików do katalogów docelowych
                        if dataset != "test":
                            # Dla zbiorów train i val kopiujemy tylko obrazy (adnotacje będą w formacie COCO)
                            target_image_path = os.path.join(dataset_dir, "images", image_name)
                            shutil.copy(source_image_path, target_image_path)
                            data_list.append(labelme_data)  # Dodaj dane do konwersji COCO
                        else:
                            # Dla zbioru test kopiujemy obrazy i oryginalne pliki JSON
                            target_image_path = os.path.join(dataset_dir, image_name)
                            target_json_path = os.path.join(dataset_dir, json_file)
                            shutil.copy(source_image_path, target_image_path)
                            shutil.copy(json_path, target_json_path)

                    except Exception as e:
                        logger.error(f"Błąd przetwarzania {json_file}: {e}")
                        continue

                # Konwersja do formatu COCO dla zbiorów train i val
                if dataset != "test" and data_list:
                    output_json_path = os.path.join(dataset_dir, "annotations", f"instances_{dataset}.json")
                    self.convert_to_coco(data_list, output_json_path)

            logger.info(f"Utworzono dataset {dataset_name} dla użytkownika {username}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd tworzenia datasetu: {e}")
            raise
            
        finally:
            # Usuń tymczasowe katalogi
            if os.path.exists(input_dir):
                shutil.rmtree(input_dir, ignore_errors=True)
                logger.debug(f"Usunięto katalog tymczasowy: {input_dir}")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                logger.debug(f"Usunięto katalog tymczasowy: {output_dir}")

    def get_dataset_info(self, username, dataset_name):
        """
        Zwraca informacje o zbiorze danych, w tym nazwy i liczbę obrazków w każdym podzbiorze.
        
        Parameters:
            username (str): Nazwa użytkownika
            dataset_name (str): Nazwa zbioru danych
            
        Returns:
            dict: Słownik zawierający informacje o każdym podzbiorze (train, val, test),
                  wraz z liczbą i nazwami obrazów
        """
        # Katalog użytkownika ze zbiorem danych
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        
        # Słownik do przechowywania informacji o zbiorze danych
        info = {}
        
        # Przetwarzanie każdego podzbioru
        for subset in ["train", "val", "test"]:
            # Uzyskaj ścieżkę do katalogu obrazów
            subset_dir = os.path.join(user_dir, subset)
            if subset in ["train", "val"]:
                subset_dir = os.path.join(subset_dir, "images")
            
            # Jeśli katalog istnieje, zbierz informacje o obrazach
            if os.path.exists(subset_dir):
                # Znajdź wszystkie pliki obrazów w katalogu
                images = [f for f in os.listdir(subset_dir) if f.lower().endswith(tuple(self.allowed_image_extensions))]
                
                # Zapisz informacje o podzbiorze
                info[subset] = {
                    "count": len(images),
                    "images": images
                }
            else:
                # Jeśli katalog nie istnieje, zwróć puste informacje
                info[subset] = {
                    "count": 0,
                    "images": []
                }
                
        logger.info(f"Pobrano informacje o datasecie {dataset_name} dla użytkownika {username}")
        return info

    def list_datasets(self, username):
        """
        Zwraca listę zbiorów danych należących do użytkownika.
        
        Parameters:
            username (str): Nazwa użytkownika
            
        Returns:
            list: Lista nazw zbiorów danych użytkownika
        """
        # Katalog użytkownika
        user_dir = os.path.join(self.output_base_dir, username)
        
        # Jeśli katalog nie istnieje, zwróć pustą listę
        if not os.path.exists(user_dir):
            return []
            
        # Znajdź wszystkie podkatalogi (każdy reprezentuje jeden zbiór danych)
        datasets = [d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))]
        
        logger.info(f"Znaleziono {len(datasets)} datasetów dla użytkownika {username}")
        return datasets

    def create_zip(self, user_dir, subset=None):
        """
        Tworzy plik ZIP dla całego zbioru danych lub wybranego podzbioru.
        
        Parameters:
            user_dir (str): Ścieżka do katalogu użytkownika ze zbiorem danych
            subset (str, optional): Nazwa podzbioru do spakowania (train, val, test).
                                    Jeśli None, pakuje cały zbiór danych.
            
        Returns:
            str: Ścieżka do utworzonego pliku ZIP
        """
        # Ścieżka do pliku ZIP
        zip_path = os.path.join(user_dir, f"{subset if subset else 'full'}_results.zip")
        
        # Lista podzbiorów do spakowania
        subsets = [subset] if subset else ["train", "val", "test"]
        
        # Tworzenie archiwum ZIP
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            # Przetwarzanie każdego podzbioru
            for subset in subsets:
                subset_dir = os.path.join(user_dir, subset)
                
                # Jeśli katalog podzbioru nie istnieje, pomiń go
                if not os.path.exists(subset_dir):
                    continue
                    
                # Przejdź przez wszystkie pliki w podzbiorze
                for root, _, files in os.walk(subset_dir):
                    for file in files:
                        # Pełna ścieżka do pliku
                        file_path = os.path.join(root, file)
                        
                        # Ścieżka w archiwum ZIP
                        arcname = os.path.join(subset, os.path.relpath(file_path, subset_dir))
                        
                        # Dla podzbiorów train i val, sprawdź czy plik w katalogu images jest obrazem
                        if subset != "test" and "images" in arcname:
                            _, ext = os.path.splitext(file.lower())
                            if ext not in self.allowed_image_extensions:
                                continue
                                
                        # Dodaj plik do archiwum
                        zipf.write(file_path, arcname)
                        
        logger.info(f"Utworzono plik ZIP: {zip_path}")
        return zip_path

    def download_dataset(self, username, dataset_name, subset=None):
        """
        Zwraca ścieżkę do pliku ZIP ze zbiorem danych lub jego podzbiorem.
        Tworzy plik ZIP, jeśli nie istnieje.
        
        Parameters:
            username (str): Nazwa użytkownika
            dataset_name (str): Nazwa zbioru danych
            subset (str, optional): Nazwa podzbioru do pobrania (train, val, test).
                                    Jeśli None, zwraca cały zbiór danych.
            
        Returns:
            str: Ścieżka do pliku ZIP zawierającego zbiór danych
            
        Raises:
            ValueError: Gdy nie udało się utworzyć pliku ZIP
        """
        # Katalog użytkownika ze zbiorem danych
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        
        # Ścieżka do pliku ZIP
        zip_path = os.path.join(user_dir, f"{subset if subset else 'full'}_results.zip")
        
        # Jeśli plik ZIP nie istnieje, utwórz go
        if not os.path.exists(zip_path):
            zip_path = self.create_zip(user_dir, subset)
        
        # Sprawdź, czy plik ZIP został utworzony
        if not os.path.exists(zip_path):
            raise ValueError(f"Nie udało się utworzyć pliku ZIP dla {'podzbioru ' + subset if subset else 'całego datasetu'}.")
        
        logger.info(f"Pobieranie pliku ZIP: {zip_path}")
        return zip_path

    def delete_dataset(self, username, dataset_name):
        """
        Usuwa zbiór danych użytkownika.
        
        Parameters:
            username (str): Nazwa użytkownika
            dataset_name (str): Nazwa zbioru danych do usunięcia
            
        Returns:
            bool: True, jeśli usunięcie powiodło się
            
        Raises:
            ValueError: Gdy zbiór danych nie istnieje lub wystąpił błąd podczas usuwania
        """
        # Katalog użytkownika ze zbiorem danych
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        
        # Sprawdź, czy zbiór danych istnieje
        if not os.path.exists(user_dir):
            raise ValueError(f"Dataset {dataset_name} nie istnieje.")
        
        try:
            # Usuń cały katalog zbioru danych
            shutil.rmtree(user_dir, ignore_errors=True)
            logger.info(f"Usunięto dataset {dataset_name} dla użytkownika {username}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas usuwania datasetu: {e}")
            raise ValueError(f"Błąd podczas usuwania datasetu: {e}")