from PyQt5 import QtWidgets, QtGui, QtCore
import requests
import os
import re
import uuid
import time
import shutil
import logging
import glob
import json
import cv2
import numpy as np
from PIL import Image
import zipfile
import base64

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, image_path=None, annotations=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.image = None
        if image_path:
            self.image = cv2.imread(image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_height = 400
        self.original_width = 600
        if self.image is not None:
            self.original_height, self.original_width = self.image.shape[:2]
        self.annotations = annotations if annotations is not None else []
        self.selected_mask_idx = -1
        self.scale = 1.0
        self.drawing = False
        self.new_box = None
        self.start_point = None
        self.setMinimumSize(600, 400)

    def resizeEvent(self, event):
        if self.image is not None:
            window_width = self.width()
            window_height = self.height()
            image_aspect = self.original_width / self.original_height
            window_aspect = window_width / window_height

            if window_aspect > image_aspect:
                self.scale = window_height / self.original_height
            else:
                self.scale = window_width / self.original_width
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        scaled_width = int(self.original_width * self.scale)
        scaled_height = int(self.original_height * self.scale)

        if self.image is None:
            painter.fillRect(0, 0, scaled_width, scaled_height, QtGui.QColor(200, 200, 200))
            painter.drawText(scaled_width // 2 - 50, scaled_height // 2, "Brak obrazu")
            return

        scaled_image = cv2.resize(self.image, (scaled_width, scaled_height))
        qimage = QtGui.QImage(scaled_image.data, scaled_image.shape[1], scaled_image.shape[0], scaled_image.strides[0], QtGui.QImage.Format_RGB888)
        painter.drawImage(0, 0, qimage)

        for idx, shape in enumerate(self.annotations):
            if shape.get("shape_type") != "mask":
                continue
            mask_base64 = shape["mask"]
            mask_data = base64.b64decode(mask_base64)
            mask_np = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask_np is None:
                continue

            points = shape["points"]
            x_min, y_min = points[0]
            x_max, y_max = points[1]

            # Oblicz wymiary docelowego obszaru po zaokrągleniu
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Przeskaluj maskę do dokładnych wymiarów obszaru docelowego
            mask_resized = cv2.resize(mask_np, (bbox_width, bbox_height), interpolation=cv2.INTER_NEAREST)

            # Upewnij się, że wymiary są zgodne
            full_mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
            target_slice = full_mask[y_min:y_max, x_min:x_max]
            if target_slice.shape != mask_resized.shape:
                # Jeśli wymiary się nie zgadzają, dostosuj mask_resized
                mask_resized = cv2.resize(mask_resized, (target_slice.shape[1], target_slice.shape[0]), interpolation=cv2.INTER_NEAREST)
            full_mask[y_min:y_max, x_min:x_max] = mask_resized

            full_mask_resized = cv2.resize(full_mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(full_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            color = (0, 255, 0, 128) if idx != self.selected_mask_idx else (255, 0, 0, 128)
            brush = QtGui.QBrush(QtGui.QColor(*color))
            painter.setBrush(brush)
            pen = QtGui.QPen(QtGui.QColor(*color[:3]), 2)
            painter.setPen(pen)
            for contour in contours:
                qpoints = [QtCore.QPoint(pt[0][0], pt[0][1]) for pt in contour]
                polygon = QtGui.QPolygon(qpoints)
                painter.drawPolygon(polygon)

        if self.drawing and self.start_point and self.new_box:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 255), 2)
            painter.setPen(pen)
            x, y, w, h = self.new_box
            painter.drawRect(x, y, w, h)

    def mousePressEvent(self, event):
        if not self.image:
            return
        if event.button() == QtCore.Qt.LeftButton:
            x = event.x() / self.scale
            y = event.y() / self.scale

            selected_idx = -1
            for idx, shape in enumerate(self.annotations):
                if shape.get("shape_type") != "mask":
                    continue
                mask_base64 = shape["mask"]
                mask_data = base64.b64decode(mask_base64)
                mask_np = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                points = shape["points"]
                x_min, y_min = points[0]
                x_max, y_max = points[1]
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                mask_resized = cv2.resize(mask_np, (int(bbox_width), int(bbox_height)), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
                target_slice = full_mask[y_min:y_max, x_min:x_max]
                if target_slice.shape != mask_resized.shape:
                    mask_resized = cv2.resize(mask_resized, (target_slice.shape[1], target_slice.shape[0]), interpolation=cv2.INTER_NEAREST)
                full_mask[y_min:y_max, x_min:x_max] = mask_resized
                if 0 <= int(x) < self.original_width and 0 <= int(y) < self.original_height and full_mask[int(y), int(x)] > 0:
                    selected_idx = idx
                    break

            if selected_idx >= 0:
                self.selected_mask_idx = selected_idx
                self.parent().update_mask_list()
                self.update()
            else:
                self.drawing = True
                self.start_point = (x, y)
                self.new_box = [x, y, 0, 0]

    def mouseMoveEvent(self, event):
        if self.drawing and self.start_point:
            end_x = event.x() / self.scale
            end_y = event.y() / self.scale
            x_min = min(self.start_point[0], end_x)
            y_min = min(self.start_point[1], end_y)
            x_max = max(self.start_point[0], end_x)
            y_max = max(self.start_point[1], end_y)
            self.new_box = [x_min, y_min, x_max - x_min, y_max - y_min]
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.drawing:
            self.drawing = False
            x_min, y_min, w, h = self.new_box
            if w > 5 and h > 5:
                mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
                x_max = x_min + w
                y_max = y_min + h
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                mask[y_min:y_max, x_min:x_max] = 255
                cropped_mask = mask[y_min:y_max, x_min:x_max]
                success, encoded_image = cv2.imencode('.png', cropped_mask)
                if success:
                    mask_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                    new_shape = {
                        "label": "pipe",
                        "points": [[float(x_min), float(y_min)], [float(x_max), float(y_max)]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "mask",
                        "flags": {},
                        "mask": mask_base64
                    }
                    self.annotations.append(new_shape)
                    self.parent().update_mask_list()
            self.new_box = None
            self.start_point = None
            self.update()

class AutoLabelingTab(QtWidgets.QWidget):
    def __init__(self, user_role):
        super().__init__()
        self.user_role = user_role
        self.api_url = "http://localhost:8000"
        self.job_name = None
        self.temp_dir = None
        self.image_paths = []
        self.annotation_files = []
        self.current_image_idx = 0
        self.annotations = []
        self.init_ui()

    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(main_widget)

        # Górna część: Wybór katalogu i modelu
        top_layout = QtWidgets.QHBoxLayout()
        self.input_dir_label = QtWidgets.QLabel("Katalog ze zdjęciami:")
        top_layout.addWidget(self.input_dir_label)
        self.input_dir_edit = QtWidgets.QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        top_layout.addWidget(self.input_dir_edit, stretch=2)
        self.input_dir_button = QtWidgets.QPushButton("Wybierz katalog")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        top_layout.addWidget(self.input_dir_button)
        self.model_version_label = QtWidgets.QLabel("Wybierz model:")
        top_layout.addWidget(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        self.update_model_versions()
        top_layout.addWidget(self.model_version_combo, stretch=1)
        self.label_button = QtWidgets.QPushButton("Uruchom automatyczne labelowanie")
        self.label_button.clicked.connect(self.run_auto_labeling)
        top_layout.addWidget(self.label_button)
        layout.addLayout(top_layout)

        # Główna część: Obraz i panele boczne
        self.main_layout = QtWidgets.QHBoxLayout()

        # Lewa strona: Podgląd obrazu
        self.image_viewer_container = QtWidgets.QWidget()
        self.image_viewer_layout = QtWidgets.QVBoxLayout()
        self.image_viewer_container.setLayout(self.image_viewer_layout)
        self.image_navigation = QtWidgets.QHBoxLayout()
        self.prev_image_btn = QtWidgets.QPushButton("Poprzedni obraz")
        self.prev_image_btn.clicked.connect(self.prev_image)
        self.next_image_btn = QtWidgets.QPushButton("Następny obraz")
        self.next_image_btn.clicked.connect(self.next_image)
        self.image_navigation.addWidget(self.prev_image_btn)
        self.image_navigation.addWidget(self.next_image_btn)
        self.image_viewer_layout.addLayout(self.image_navigation)
        self.image_viewer = ImageViewer(parent=self)
        self.image_viewer_layout.insertWidget(0, self.image_viewer)
        self.main_layout.addWidget(self.image_viewer_container, stretch=4)

        # Prawa strona: Lista masek i lista obrazów
        self.right_panel = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout()
        
        # Lista masek
        self.mask_list_widget = QtWidgets.QWidget()
        self.mask_list_layout = QtWidgets.QVBoxLayout()
        self.mask_list_label = QtWidgets.QLabel("Lista masek:")
        self.mask_list_layout.addWidget(self.mask_list_label)
        self.mask_list = QtWidgets.QListWidget()
        self.mask_list.itemClicked.connect(self.select_mask)
        self.mask_list_layout.addWidget(self.mask_list)
        self.delete_mask_btn = QtWidgets.QPushButton("Usuń wybraną maskę")
        self.delete_mask_btn.clicked.connect(self.delete_mask)
        self.mask_list_layout.addWidget(self.delete_mask_btn)
        self.save_annotations_btn = QtWidgets.QPushButton("Zapisz zmiany")
        self.save_annotations_btn.clicked.connect(self.save_annotations)
        self.mask_list_layout.addWidget(self.save_annotations_btn)
        self.download_btn = QtWidgets.QPushButton("Pobierz wyniki")
        self.download_btn.clicked.connect(self.download_results)
        self.download_btn.setEnabled(False)
        self.mask_list_layout.addWidget(self.download_btn)
        self.mask_list_widget.setLayout(self.mask_list_layout)
        self.right_layout.addWidget(self.mask_list_widget)

        # Lista obrazów (w prawym dolnym rogu, jak w LabelMe)
        self.image_list_widget = QtWidgets.QWidget()
        self.image_list_layout = QtWidgets.QVBoxLayout()
        self.image_list_label = QtWidgets.QLabel("Lista obrazów:")
        self.image_list_layout.addWidget(self.image_list_label)
        self.image_list = QtWidgets.QListWidget()
        self.image_list.itemClicked.connect(self.select_image)
        self.image_list_layout.addWidget(self.image_list)
        self.image_list_widget.setLayout(self.image_list_layout)
        self.right_layout.addWidget(self.image_list_widget)

        self.right_panel.setLayout(self.right_layout)
        self.main_layout.addWidget(self.right_panel, stretch=1)

        layout.addLayout(self.main_layout)

        # Pasek przewijania
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def select_input_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz katalog ze zdjęciami")
        if directory:
            self.input_dir_edit.setText(directory)

    def update_model_versions(self):
        try:
            logger.debug("Pobieram listę modeli z %s/model_versions", self.api_url)
            response = requests.get(f"{self.api_url}/model_versions_maskrcnn")
            response.raise_for_status()
            model_versions = response.json()
            self.model_version_combo.clear()
            if model_versions:
                self.model_version_combo.addItems(model_versions)
            else:
                self.model_version_combo.addItem("Brak dostępnych modeli")
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas pobierania modeli: %s", e)
            self.model_version_combo.clear()
            self.model_version_combo.addItem("Brak dostępnych modeli")

    def run_auto_labeling(self):
        input_dir = self.input_dir_edit.text()
        model_version = self.model_version_combo.currentText()

        if not input_dir:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać katalog ze zdjęciami!")
            return
        if not model_version or model_version == "Brak dostępnych modeli":
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać model!")
            return

        self.job_name = f"auto_label_{uuid.uuid4().hex}"

        image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
        if not image_paths:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Brak obrazów .jpg w katalogu!")
            return

        try:
            logger.debug("Wysyłam żądanie do %s/auto_label: job_name=%s, model_version=%s, %d obrazów",
                         self.api_url, self.job_name, model_version, len(image_paths))
            files = [('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in image_paths]
            data = {'job_name': self.job_name, 'model_version': model_version}
            response = requests.post(
                f"{self.api_url}/auto_label",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            if result["status"] != "success":
                QtWidgets.QMessageBox.warning(self, "Błąd", result.get('message', 'Nieznany błąd'))
                return

            # Pobierz wyniki
            self.temp_dir = f"temp_{self.job_name}"
            os.makedirs(self.temp_dir, exist_ok=True)
            zip_path = os.path.join(self.temp_dir, f"{self.job_name}_results.zip")
            response = requests.get(f"{self.api_url}/get_results/{self.job_name}")
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)

            # Rozpakuj ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            self.image_paths = glob.glob(os.path.join(self.temp_dir, f"{self.job_name}_after", "*.jpg"))
            self.annotation_files = glob.glob(os.path.join(self.temp_dir, f"{self.job_name}_after", "*.json"))
            self.image_paths.sort()
            self.annotation_files.sort()

            if not self.image_paths or not self.annotation_files:
                QtWidgets.QMessageBox.warning(self, "Błąd", "Brak wyników auto-labelingu!")
                return

            self.current_image_idx = 0
            self.load_current_image()
            self.update_image_list()
            self.download_btn.setEnabled(True)
        except requests.exceptions.RequestException as e:
            logger.error("Błąd podczas labelowania: %s", e)
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas labelowania: {e}")
        finally:
            for _, file_tuple in files:
                file_tuple[1].close()

    def load_current_image(self):
        if not self.image_paths:
            return
        image_path = self.image_paths[self.current_image_idx]
        annotation_path = self.annotation_files[self.current_image_idx]

        with open(annotation_path, 'r') as f:
            json_data = json.load(f)
        self.annotations = json_data["shapes"]

        if self.image_viewer:
            self.image_viewer_layout.removeWidget(self.image_viewer)
            self.image_viewer.deleteLater()
        self.image_viewer = ImageViewer(image_path, self.annotations, self)
        self.image_viewer_layout.insertWidget(0, self.image_viewer)
        self.update_mask_list()
        self.image_list.setCurrentRow(self.current_image_idx)

    def update_mask_list(self):
        self.mask_list.clear()
        for idx, shape in enumerate(self.annotations):
            self.mask_list.addItem(f"Maska {idx + 1}: {shape['label']}")
        if self.image_viewer.selected_mask_idx >= 0:
            self.mask_list.setCurrentRow(self.image_viewer.selected_mask_idx)

    def update_image_list(self):
        self.image_list.clear()
        for image_path in self.image_paths:
            self.image_list.addItem(os.path.basename(image_path))

    def select_mask(self, item):
        idx = self.mask_list.row(item)
        self.image_viewer.selected_mask_idx = idx
        self.image_viewer.update()

    def select_image(self, item):
        idx = self.image_list.row(item)
        self.current_image_idx = idx
        self.load_current_image()

    def delete_mask(self):
        if self.image_viewer.selected_mask_idx >= 0:
            self.annotations.pop(self.image_viewer.selected_mask_idx)
            self.image_viewer.selected_mask_idx = -1
            self.update_mask_list()
            self.image_viewer.update()

    def save_annotations(self):
        if not self.annotation_files:
            return
        annotation_path = self.annotation_files[self.current_image_idx]
        json_data = {
            "version": "5.8.1",
            "flags": {},
            "shapes": self.annotations,
            "imagePath": os.path.basename(self.image_paths[self.current_image_idx]),
            "imageData": None,
            "imageHeight": self.image_viewer.original_height,
            "imageWidth": self.image_viewer.original_width
        }
        with open(annotation_path, 'w') as f:
            json.dump(json_data, f, indent=2)

    def download_results(self):
        if not self.job_name or not self.temp_dir:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Brak wyników do pobrania!")
            return

        zip_path = os.path.join(self.temp_dir, f"{self.job_name}_results.zip")
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Zapisz wyniki", f"{self.job_name}_results.zip", "ZIP files (*.zip)"
        )
        if save_path:
            shutil.copy(zip_path, save_path)
            # Opcjonalnie usuń foldery tymczasowe
            # shutil.rmtree(self.temp_dir, ignore_errors=True)

    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_image_idx < len(self.image_paths) - 1:
            self.current_image_idx += 1
            self.load_current_image()