import os
import json
import base64
import shutil
import zipfile
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from pycocotools import mask as mask_utils
from io import BytesIO
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DatasetAPI:
    def __init__(self):
        self.train_ratio = 0.7
        self.output_base_dir = "/app/backend/data/Dataset_creation"

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

    def create_dataset(self, job_name, input_files):
        """Tworzy dataset z przesłanych plików i zwraca .zip."""
        input_dir = f"{self.output_base_dir}/{job_name}_input"
        output_dir = f"{self.output_base_dir}/{job_name}_output"
        zip_path = f"{self.output_base_dir}/{job_name}_results.zip"

        try:
            # Utwórz katalogi
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Zapisz przesłane pliki
            json_files = []
            for file in input_files:
                file_path = os.path.join(input_dir, file.filename)
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                if file.filename.endswith(".json"):
                    json_files.append(file.filename)

            if not json_files:
                raise ValueError("Brak plików .json w przesłanych danych.")

            # Podziel na train i val
            train_files, val_files = train_test_split(json_files, train_size=self.train_ratio, random_state=42)

            datasets = {
                "train": train_files,
                "val": val_files
            }

            for dataset, files in datasets.items():
                dataset_dir = f"{output_dir}/{dataset}"
                os.makedirs(f"{dataset_dir}/images", exist_ok=True)
                os.makedirs(f"{dataset_dir}/annotations", exist_ok=True)

                data_list = []
                for json_file in files:
                    json_path = os.path.join(input_dir, json_file)
                    try:
                        labelme_data = self.load_labelme_json(json_path)
                        image_name = labelme_data["imagePath"]
                        source_image_path = os.path.join(input_dir, image_name)

                        if not os.path.exists(source_image_path):
                            logger.warning("Brak obrazu %s, pomijam.", source_image_path)
                            continue

                        target_image_path = f"{dataset_dir}/images/{image_name}"
                        shutil.copy(source_image_path, target_image_path)
                        data_list.append(labelme_data)
                    except Exception as e:
                        logger.error("Błąd przetwarzania %s: %s", json_file, e)
                        continue

                if data_list:
                    output_json_path = f"{dataset_dir}/annotations/instances_{dataset}.json"
                    self.convert_to_coco(data_list, output_json_path)
                else:
                    logger.warning("Brak danych dla %s, pomijam COCO JSON.", dataset)

            # Spakuj wyniki
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
                        logger.debug("Dodano do zip: %s", arcname)

            return zip_path
        except Exception as e:
            logger.error("Błąd tworzenia datasetu: %s", e)
            raise
        finally:
            # Usuń tymczasowe katalogi w tle
            if os.path.exists(input_dir):
                shutil.rmtree(input_dir, ignore_errors=True)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)