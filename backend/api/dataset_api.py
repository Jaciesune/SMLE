import os
import json
import base64
import shutil
import zipfile
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from io import BytesIO
import logging
import random

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAPI:
    def __init__(self):
        self.output_base_dir = "/app/backend/data/dataset_create"
        self.allowed_image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    def decode_mask(self, base64_str):
        """Dekoduje maskę zapisaną w Base64 do postaci NumPy."""
        try:
            mask_bytes = base64.b64decode(base64_str)
            mask_image = Image.open(BytesIO(mask_bytes)).convert("L")
            return np.array(mask_image, dtype=np.uint8)
        except Exception as e:
            logger.error("Błąd dekodowania maski: %s", e)
            raise ValueError(f"Błąd dekodowania maski: {e}")

    def normalize_image_path(self, image_path):
        """Normalizuje ścieżkę obrazu do samej nazwy pliku."""
        return os.path.basename(image_path)

    def load_labelme_json(self, json_path):
        """Wczytuje plik JSON z LabelMe i normalizuje imagePath."""
        with open(json_path, "r") as f:
            data = json.load(f)
        data["imagePath"] = self.normalize_image_path(data["imagePath"])
        return data

    def encode_rle(self, mask):
        """Koduje maskę binarną do formatu RLE."""
        mask = np.asfortranarray(mask)
        return mask_utils.encode(mask)

    def convert_to_coco(self, data_list, output_path):
        """Konwertuje LabelMe do COCO JSON z maskami w formacie RLE."""
        coco = {
            "info": {"description": "Dataset converted from LabelMe"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        category_map = {}
        annotation_id = 1

        for image_id, data in enumerate(data_list):
            coco["images"].append({
                "id": image_id,
                "file_name": data["imagePath"],
                "width": data["imageWidth"],
                "height": data["imageHeight"]
            })

            for shape in data["shapes"]:
                label = shape["label"]
                if label not in category_map:
                    category_map[label] = len(category_map) + 1
                    coco["categories"].append({"id": category_map[label], "name": label})

                bbox = shape["points"]
                try:
                    xmin, ymin = map(int, bbox[0])
                    xmax, ymax = map(int, bbox[1])
                    width, height = xmax - xmin, ymax - ymin
                except Exception as e:
                    logger.error("Błąd w bbox dla %s: %s", data["imagePath"], e)
                    continue

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[label],
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0
                }

                if "mask" in shape:
                    try:
                        mask_array = self.decode_mask(shape["mask"])
                        rle = self.encode_rle(mask_array)
                        annotation["segmentation"] = {
                            "counts": rle["counts"].decode("utf-8"),
                            "size": list(rle["size"])
                        }
                    except ValueError as e:
                        logger.warning("Pomijam maskę dla %s: %s", data["imagePath"], e)
                        continue

                coco["annotations"].append(annotation)
                annotation_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=4)

    def check_duplicates(self, username, dataset_name, image_name):
        """Sprawdza, czy obraz już istnieje w którejś z kategorii."""
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        for subset in ["train", "val", "test"]:
            subset_dir = os.path.join(user_dir, subset)
            if subset in ["train", "val"]:
                subset_dir = os.path.join(subset_dir, "images")
            if os.path.exists(subset_dir) and image_name in os.listdir(subset_dir):
                return True
        return False

    def manual_split(self, files, train_ratio, val_ratio, test_ratio):
        """Ręcznie dzieli listę plików na train, val i test według podanych proporcji w losowy sposób."""
        total = len(files)
        if total == 0:
            return [], [], []

        random.shuffle(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        test_end = total

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:test_end]

        return train_files, val_files, test_files

    def create_dataset(self, username, dataset_name, input_files, train_ratio, val_ratio, test_ratio):
        """Tworzy dataset z przesłanych plików bez tworzenia ZIP."""
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        input_dir = os.path.join(user_dir, "input")
        output_dir = os.path.join(user_dir, "output")

        try:
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            json_files = []
            image_files = []
            for file in input_files:
                file_path = os.path.join(input_dir, file.filename)
                _, ext = os.path.splitext(file.filename.lower())
                if ext in self.allowed_image_extensions:
                    with open(file_path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                    image_files.append(file.filename)
                elif ext == ".json":
                    with open(file_path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                    json_files.append(file.filename)
                else:
                    logger.warning("Pomijam plik o nieobsługiwanym rozszerzeniu: %s", file.filename)

            if not json_files:
                raise ValueError("Brak plików .json w przesłanych danych.")

            if not image_files:
                raise ValueError("Brak plików obrazów (.jpg, .jpeg, .png, .bmp) w przesłanych danych.")

            paired_files = []
            for img in image_files:
                json_name = img.rsplit(".", 1)[0] + ".json"
                if json_name not in json_files:
                    logger.warning("Brak pliku JSON dla obrazu %s, pomijam.", img)
                    continue
                if self.check_duplicates(username, dataset_name, img):
                    logger.warning("Obraz %s już istnieje w datasecie, pomijam.", img)
                    continue
                paired_files.append(img)

            if not paired_files:
                raise ValueError("Brak nowych par obraz-JSON do przetworzenia.")

            train_files, val_files, test_files = self.manual_split(
                paired_files, train_ratio, val_ratio, test_ratio
            )

            datasets = {
                "train": train_files,
                "val": val_files,
                "test": test_files
            }

            for dataset, files in datasets.items():
                dataset_dir = os.path.join(user_dir, dataset)
                if dataset != "test":
                    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
                    os.makedirs(os.path.join(dataset_dir, "annotations"), exist_ok=True)
                else:
                    os.makedirs(dataset_dir, exist_ok=True)

                data_list = []
                for img_file in files:
                    json_file = img_file.rsplit(".", 1)[0] + ".json"
                    json_path = os.path.join(input_dir, json_file)
                    try:
                        labelme_data = self.load_labelme_json(json_path)
                        image_name = labelme_data["imagePath"]
                        _, ext = os.path.splitext(image_name.lower())
                        if ext not in self.allowed_image_extensions:
                            logger.warning("Plik %s nie jest obrazem (rozszerzenie %s), pomijam.", image_name, ext)
                            continue

                        source_image_path = os.path.join(input_dir, image_name)
                        if not os.path.exists(source_image_path):
                            logger.warning("Brak obrazu %s, pomijam.", source_image_path)
                            continue

                        if dataset != "test":
                            target_image_path = os.path.join(dataset_dir, "images", image_name)
                            shutil.copy(source_image_path, target_image_path)
                            data_list.append(labelme_data)
                        else:
                            target_image_path = os.path.join(dataset_dir, image_name)
                            target_json_path = os.path.join(dataset_dir, json_file)
                            shutil.copy(source_image_path, target_image_path)
                            shutil.copy(json_path, target_json_path)

                    except Exception as e:
                        logger.error("Błąd przetwarzania %s: %s", json_file, e)
                        continue

                if dataset != "test" and data_list:
                    output_json_path = os.path.join(dataset_dir, "annotations", f"instances_{dataset}.json")
                    self.convert_to_coco(data_list, output_json_path)

            return True
        except Exception as e:
            logger.error("Błąd tworzenia datasetu: %s", e)
            raise
        finally:
            if os.path.exists(input_dir):
                shutil.rmtree(input_dir, ignore_errors=True)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

    def get_dataset_info(self, username, dataset_name):
        """Zwraca informacje o datasecie, w tym nazwy obrazków."""
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        info = {}
        for subset in ["train", "val", "test"]:
            subset_dir = os.path.join(user_dir, subset)
            if subset in ["train", "val"]:
                subset_dir = os.path.join(subset_dir, "images")
            if os.path.exists(subset_dir):
                images = [f for f in os.listdir(subset_dir) if f.lower().endswith(tuple(self.allowed_image_extensions))]
                info[subset] = {
                    "count": len(images),
                    "images": images
                }
            else:
                info[subset] = {
                    "count": 0,
                    "images": []
                }
        return info

    def list_datasets(self, username):
        """Zwraca listę datasetów użytkownika."""
        user_dir = os.path.join(self.output_base_dir, username)
        if not os.path.exists(user_dir):
            return []
        return [d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))]

    def create_zip(self, user_dir, subset=None):
        """Tworzy plik ZIP dla całego datasetu lub podzbioru."""
        zip_path = os.path.join(user_dir, f"{subset if subset else 'full'}_results.zip")
        subsets = [subset] if subset else ["train", "val", "test"]
        
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for subset in subsets:
                subset_dir = os.path.join(user_dir, subset)
                if not os.path.exists(subset_dir):
                    continue
                for root, _, files in os.walk(subset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(subset, os.path.relpath(file_path, subset_dir))
                        if subset != "test" and "images" in arcname:
                            _, ext = os.path.splitext(file.lower())
                            if ext not in self.allowed_image_extensions:
                                continue
                        zipf.write(file_path, arcname)
        return zip_path

    def download_dataset(self, username, dataset_name, subset=None):
        """Zwraca ZIP z danymi datasetu lub podzbioru, tworząc go jeśli nie istnieje."""
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        zip_path = os.path.join(user_dir, f"{subset if subset else 'full'}_results.zip")
        
        if not os.path.exists(zip_path):
            zip_path = self.create_zip(user_dir, subset)
        
        if not os.path.exists(zip_path):
            raise ValueError(f"Nie udało się utworzyć pliku ZIP dla {'podzbioru ' + subset if subset else 'całego datasetu'}.")
        
        return zip_path

    def delete_dataset(self, username, dataset_name):
        """Usuwa dataset użytkownika."""
        user_dir = os.path.join(self.output_base_dir, username, dataset_name)
        if not os.path.exists(user_dir):
            raise ValueError(f"Dataset {dataset_name} nie istnieje.")
        
        try:
            shutil.rmtree(user_dir, ignore_errors=True)
            return True
        except Exception as e:
            logger.error("Błąd podczas usuwania datasetu: %s", e)
            raise ValueError(f"Błąd podczas usuwania datasetu: {e}")