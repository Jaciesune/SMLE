from PyQt5 import QtWidgets, QtGui, QtCore
import requests
import os
import glob
import json
import cv2
import numpy as np
import base64
import shutil
import uuid
import zipfile
import logging
import sys
from PIL import Image  # Dodajemy Pillow do obsługi RGBA

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, image_path=None, annotations=None, auto_labeling_tab=None):
        super().__init__(auto_labeling_tab)
        self.auto_labeling_tab = auto_labeling_tab
        self.image_path = image_path
        self.image = None
        if image_path:
            try:
                image_path = os.path.normpath(image_path)
                pil_image = Image.open(image_path)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                self.image = np.array(pil_image)
                if self.image is None:
                    logger.error(f"Nie można wczytać obrazu: {image_path}")
                    raise ValueError(f"Nie można wczytać obrazu: {image_path}")
            except Exception as e:
                logger.error(f"Błąd wczytywania obrazu {image_path}: {e}")
                raise
        self.original_height = 400
        self.original_width = 600
        if self.image is not None:
            self.original_height, self.original_width = self.image.shape[:2]
        self.annotations = annotations if annotations is not None else []
        self.selected_mask_idx = -1
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.drawing = False
        self.editing = False
        self.current_polygon = []
        self.editing_point_idx = -1
        self.setMinimumSize(600, 400)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def resizeEvent(self, event):
        if self.image is not None:
            window_width = self.width()
            window_height = self.height()
            image_aspect = self.original_width / self.original_height
            window_aspect = window_width / window_height

            if window_aspect > image_aspect:
                base_scale = window_height / self.original_height
            else:
                base_scale = window_width / self.original_width
            self.update()

    def wheelEvent(self, event):
        if self.image is None:
            return
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            new_scale = self.scale * zoom_factor
        else:
            new_scale = self.scale / zoom_factor
        self.scale = max(self.min_scale, min(self.max_scale, new_scale))
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
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            mask_resized = cv2.resize(mask_np, (bbox_width, bbox_height), interpolation=cv2.INTER_NEAREST)
            full_mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
            target_slice = full_mask[y_min:y_max, x_min:x_max]
            if target_slice.shape != mask_resized.shape:
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

        if self.drawing and self.current_polygon:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 255), 2)
            painter.setPen(pen)
            qpoints = [QtCore.QPoint(int(pt[0] * self.scale), int(pt[1] * self.scale)) for pt in self.current_polygon]
            for i in range(len(qpoints) - 1):
                painter.drawLine(qpoints[i], qpoints[i + 1])
            for pt in qpoints:
                painter.drawEllipse(pt, 3, 3)

    def mousePressEvent(self, event):
        if self.image is None:
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
                mask_resized = cv2.resize(mask_np, (bbox_width, bbox_height), interpolation=cv2.INTER_NEAREST)
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
                self.auto_labeling_tab.update_mask_list()
                self.update()
            else:
                if not self.drawing:
                    self.drawing = True
                    self.current_polygon = []
                if self.drawing:
                    self.current_polygon.append((x, y))
                    self.update()

    def mouseMoveEvent(self, event):
        if self.image is None:
            return
        if self.drawing:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.editing_point_idx = -1

    def mouseDoubleClickEvent(self, event):
        if self.drawing and len(self.current_polygon) >= 3:
            label = self.auto_labeling_tab.get_current_label()
            if not label:
                QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wpisać etykietę!")
                self.drawing = False
                self.current_polygon = []
                self.update()
                return

            x_coords = [pt[0] for pt in self.current_polygon]
            y_coords = [pt[1] for pt in self.current_polygon]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
            points = np.array([(int(pt[0]), int(pt[1])) for pt in self.current_polygon], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            success, encoded_image = cv2.imencode('.png', cropped_mask)
            if success:
                mask_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                new_shape = {
                    "label": label,
                    "points": [[float(x_min), float(y_min)], [float(x_max), float(y_max)]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "mask",
                    "flags": {},
                    "mask": mask_base64
                }
                self.annotations.append(new_shape)
                self.auto_labeling_tab.update_mask_list()
                self.auto_labeling_tab.add_label_to_list(label)

            self.drawing = False
            self.current_polygon = []
            self.update()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete and self.selected_mask_idx >= 0:
            self.auto_labeling_tab.delete_mask()
        elif event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_S:
            self.auto_labeling_tab.save_annotations()

class AutoLabelingTab(QtWidgets.QWidget):
    def __init__(self, user_role):
        super().__init__()
        self.user_role = user_role
        self.api_url = "http://localhost:8000"
        self.job_name = None
        self.temp_dir = None
        self.original_image_paths = []
        self.image_paths = []
        self.annotation_files = []
        self.current_image_idx = 0
        self.annotations = []
        self.labels = ["obiekt"]
        self.init_ui()

    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(main_widget)

        self.top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(self.top_widget)

        input_layout = QtWidgets.QHBoxLayout()
        self.input_dir_label = QtWidgets.QLabel("Katalog ze zdjęciami:")
        input_layout.addWidget(self.input_dir_label)
        self.input_dir_edit = QtWidgets.QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        input_layout.addWidget(self.input_dir_edit, stretch=2)
        self.input_dir_button = QtWidgets.QPushButton("Wybierz katalog")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        input_layout.addWidget(self.input_dir_button)
        top_layout.addLayout(input_layout)

        mode_layout = QtWidgets.QHBoxLayout()
        self.mode_label = QtWidgets.QLabel("Tryb pracy:")
        mode_layout.addWidget(self.mode_label)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Ręczne oznaczanie", "Automatyczne oznaczanie"])
        mode_layout.addWidget(self.mode_combo)
        self.model_version_label = QtWidgets.QLabel("Wybierz model (tylko dla auto-labelingu):")
        mode_layout.addWidget(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        mode_layout.addWidget(self.model_version_combo)

        # Dodajemy pole do wpisywania etykiety dla modelu
        self.custom_label_label = QtWidgets.QLabel("Etykieta dla modelu:")
        mode_layout.addWidget(self.custom_label_label)
        self.custom_label_input = QtWidgets.QLineEdit()
        self.custom_label_input.setPlaceholderText("Wpisz etykietę (np. plank, pipe)")
        mode_layout.addWidget(self.custom_label_input)

        self.load_button = QtWidgets.QPushButton("Wczytaj i oznacz")
        self.load_button.clicked.connect(self.load_and_label)
        mode_layout.addWidget(self.load_button)
        top_layout.addLayout(mode_layout)

        layout.addWidget(self.top_widget)

        self.toggle_top_btn = QtWidgets.QPushButton("Pokaż/Ukryj opcje")
        self.toggle_top_btn.clicked.connect(self.toggle_top_panel)
        layout.addWidget(self.toggle_top_btn)

        self.main_layout = QtWidgets.QHBoxLayout()

        self.image_viewer_container = QtWidgets.QWidget()
        self.image_viewer_layout = QtWidgets.QVBoxLayout()
        self.image_viewer_container.setLayout(self.image_viewer_layout)
        self.image_viewer = ImageViewer(auto_labeling_tab=self)
        self.image_viewer_layout.insertWidget(0, self.image_viewer)
        self.main_layout.addWidget(self.image_viewer_container, stretch=4)

        self.right_panel = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout()

        self.label_widget = QtWidgets.QWidget()
        self.label_layout = QtWidgets.QHBoxLayout()
        self.label_input_label = QtWidgets.QLabel("Etykieta:")
        self.label_layout.addWidget(self.label_input_label)
        self.label_input = QtWidgets.QComboBox()
        self.label_input.setEditable(True)
        self.label_input.addItems(self.labels)
        self.label_layout.addWidget(self.label_input)
        self.label_widget.setLayout(self.label_layout)
        self.right_layout.addWidget(self.label_widget)

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
        self.change_label_btn = QtWidgets.QPushButton("Zmień etykietę")
        self.change_label_btn.clicked.connect(self.change_label)
        self.mask_list_layout.addWidget(self.change_label_btn)
        self.save_annotations_btn = QtWidgets.QPushButton("Zapisz zmiany (Ctrl+S)")
        self.save_annotations_btn.clicked.connect(self.save_annotations)
        self.mask_list_layout.addWidget(self.save_annotations_btn)
        self.download_btn = QtWidgets.QPushButton("Pobierz wyniki")
        self.download_btn.clicked.connect(self.download_results)
        self.download_btn.setEnabled(False)
        self.mask_list_layout.addWidget(self.download_btn)
        self.mask_list_widget.setLayout(self.mask_list_layout)
        self.right_layout.addWidget(self.mask_list_widget)

        self.image_list_widget = QtWidgets.QWidget()
        self.image_list_layout = QtWidgets.QVBoxLayout()
        self.image_list_label = QtWidgets.QLabel("Lista obrazów:")
        self.image_list_layout.addWidget(self.image_list_label)
        self.image_list = QtWidgets.QListWidget()
        self.image_list.itemClicked.connect(self.select_image)
        self.image_list_layout.addWidget(self.image_list)
        self.image_navigation = QtWidgets.QHBoxLayout()
        self.prev_image_btn = QtWidgets.QPushButton("Poprzedni obraz")
        self.prev_image_btn.clicked.connect(self.prev_image)
        self.next_image_btn = QtWidgets.QPushButton("Następny obraz")
        self.next_image_btn.clicked.connect(self.next_image)
        self.image_navigation.addWidget(self.prev_image_btn)
        self.image_navigation.addWidget(self.next_image_btn)
        self.image_list_layout.addLayout(self.image_navigation)
        self.image_list_widget.setLayout(self.image_list_layout)
        self.right_layout.addWidget(self.image_list_widget)

        self.right_panel.setLayout(self.right_layout)
        self.main_layout.addWidget(self.right_panel, stretch=1)

        layout.addLayout(self.main_layout)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        self.update_model_versions()

    def toggle_top_panel(self):
        self.top_widget.setVisible(not self.top_widget.isVisible())

    def select_input_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz katalog ze zdjęciami")
        if directory:
            self.input_dir_edit.setText(directory)

    def update_model_versions(self):
        try:
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
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas pobierania modeli: {e}")
            self.model_version_combo.clear()
            self.model_version_combo.addItem("Brak dostępnych modeli")

    def find_results_dir(self, base_path, job_name):
        target_dir = f"{job_name}_after"
        for root, dirs, _ in os.walk(base_path):
            if target_dir in dirs:
                found_dir = os.path.join(root, target_dir)
                logger.debug(f"Znalazłem katalog wyników: {found_dir}")
                return found_dir
        logger.error(f"Nie znaleziono katalogu {target_dir} w {base_path}")
        return None

    def log_directory_structure(self, base_path):
        logger.debug(f"Struktura katalogu {base_path}:")
        for root, dirs, files in os.walk(base_path):
            level = root.replace(base_path, "").count(os.sep)
            indent = " " * 4 * level
            logger.debug(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                logger.debug(f"{subindent}{f}")

    def load_and_label(self):
        input_dir = self.input_dir_edit.text()
        mode = self.mode_combo.currentText()

        if not input_dir:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać katalog ze zdjęciami!")
            return

        self.original_image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
        if not self.original_image_paths:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Brak obrazów .jpg w katalogu!")
            return

        self.original_image_paths.sort()
        self.image_paths = self.original_image_paths.copy()
        self.annotation_files = []

        for image_path in self.original_image_paths:
            base_name = os.path.splitext(image_path)[0]
            annotation_path = f"{base_name}.json"
            if os.path.exists(annotation_path):
                self.annotation_files.append(annotation_path)
            else:
                json_data = {
                    "version": "5.8.1",
                    "flags": {},
                    "shapes": [],
                    "imagePath": os.path.basename(image_path),
                    "imageData": None,
                    "imageHeight": 0,
                    "imageWidth": 0
                }
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                self.annotation_files.append(annotation_path)

        if mode == "Automatyczne oznaczanie":
            model_version = self.model_version_combo.currentText()
            custom_label = self.custom_label_input.text().strip()

            if not model_version or model_version == "Brak dostępnych modeli":
                QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wybrać model!")
                return

            if not custom_label:
                QtWidgets.QMessageBox.warning(self, "Błąd", "Proszę wpisać etykietę dla modelu!")
                return

            self.job_name = f"auto_label_{uuid.uuid4().hex}"
            try:
                files = [('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in self.original_image_paths]
                data = {
                    'job_name': self.job_name,
                    'model_version': model_version,
                    'custom_label': custom_label  # Przekazujemy etykietę użytkownika
                }
                logger.debug(f"Wysyłam żądanie do /auto_label z job_name={self.job_name}, custom_label={custom_label}")
                response = requests.post(
                    f"{self.api_url}/auto_label",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                result = response.json()
                logger.debug(f"Otrzymano odpowiedź z /auto_label: {result}")
                if result["status"] != "success":
                    QtWidgets.QMessageBox.warning(self, "Błąd", result.get('message', 'Nieznany błąd'))
                    return

                # Sprawdź, czy odpowiedź zawiera informację o braku wyników
                if "message" in result and "Nie znaleziono obiektów" in result["message"]:
                    QtWidgets.QMessageBox.information(self, "Informacja", "Auto-labeling nie znalazł żadnych obiektów. Możesz przejść do ręcznego oznaczania.")
                    # Wczytaj oryginalne obrazy do ręcznego oznaczania
                    self.current_image_idx = 0
                    self.load_current_image()
                    self.update_image_list()
                    self.download_btn.setEnabled(True)
                    self.top_widget.setVisible(False)
                    return

                # Zmień katalog zapisu na backend/data
                data_dir = os.path.join(os.getcwd(), "backend", "data")
                os.makedirs(data_dir, exist_ok=True)

                self.temp_dir = os.path.join(data_dir, f"temp_{self.job_name}")
                os.makedirs(self.temp_dir, exist_ok=True)

                logger.debug(f"Pobieram wyniki z /get_results/{self.job_name}")
                response = requests.get(f"{self.api_url}/get_results/{self.job_name}")
                response.raise_for_status()
                zip_path = os.path.join(self.temp_dir, f"{self.job_name}_results.zip")
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                logger.debug(f"Zapisano ZIP do {zip_path}")

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
                logger.debug(f"Rozpakowano ZIP do {self.temp_dir}")

                self.log_directory_structure(self.temp_dir)

                os.remove(zip_path)
                logger.debug(f"Usunięto ZIP: {zip_path}")

                after_dir = self.find_results_dir(self.temp_dir, self.job_name)
                if not after_dir:
                    logger.error(f"Katalog z wynikami auto-labelingu nie istnieje w {self.temp_dir}!")
                    QtWidgets.QMessageBox.warning(self, "Błąd", "Katalog z wynikami auto-labelingu nie istnieje!")
                    return

                self.image_paths = glob.glob(os.path.join(after_dir, "*.jpg"))
                self.annotation_files = glob.glob(os.path.join(after_dir, "*.json"))
                self.image_paths.sort()
                self.annotation_files.sort()

                if not self.image_paths or not self.annotation_files:
                    logger.error("Brak plików obrazów lub adnotacji w katalogu wyników!")
                    QtWidgets.QMessageBox.warning(self, "Błąd", "Brak wyników auto-labelingu (obrazów lub adnotacji)!")
                    return

                if len(self.image_paths) != len(self.annotation_files):
                    logger.error(f"Niezgodność liczby obrazów ({len(self.image_paths)}) i adnotacji ({len(self.annotation_files)})!")
                    QtWidgets.QMessageBox.warning(self, "Błąd", "Liczba obrazów i adnotacji nie jest zgodna!")
                    return

            except requests.exceptions.RequestException as e:
                logger.error("Błąd podczas labelowania: %s", e)
                QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas labelowania: {e}")
                return
            finally:
                for _, file_tuple in files:
                    file_tuple[1].close()

        self.current_image_idx = 0
        self.load_current_image()
        self.update_image_list()
        self.download_btn.setEnabled(True)
        self.top_widget.setVisible(False)

    def load_current_image(self):
        if not self.image_paths or not self.annotation_files:
            logger.error("Brak ścieżek do obrazów lub adnotacji!")
            return

        if self.current_image_idx >= len(self.image_paths) or self.current_image_idx >= len(self.annotation_files):
            logger.error(f"Nieprawidłowy indeks obrazu: {self.current_image_idx}")
            return

        image_path = self.image_paths[self.current_image_idx]
        annotation_path = self.annotation_files[self.current_image_idx]

        if not os.path.exists(image_path):
            logger.error(f"Plik obrazu nie istnieje: {image_path}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Plik obrazu nie istnieje: {image_path}")
            return
        if not os.path.exists(annotation_path):
            logger.error(f"Plik adnotacji nie istnieje: {annotation_path}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Plik adnotacji nie istnieje: {annotation_path}")
            return

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            self.annotations = json_data["shapes"]
        except Exception as e:
            logger.error(f"Błąd wczytywania adnotacji {annotation_path}: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd wczytywania adnotacji: {e}")
            return

        if hasattr(self, 'image_viewer') and self.image_viewer is not None:
            self.image_viewer_layout.removeWidget(self.image_viewer)
            self.image_viewer.deleteLater()
            self.image_viewer = None

        try:
            self.image_viewer = ImageViewer(image_path, self.annotations, auto_labeling_tab=self)
        except Exception as e:
            logger.error(f"Błąd wczytywania obrazu {image_path}: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd wczytywania obrazu: {e}")
            return

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
        for i, (image_path, annotation_path) in enumerate(zip(self.image_paths, self.annotation_files)):
            item = QtWidgets.QListWidgetItem(os.path.basename(image_path))
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                has_annotations = len(json_data["shapes"]) > 0
            except Exception:
                has_annotations = False
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if has_annotations else QtCore.Qt.Unchecked)
            self.image_list.addItem(item)

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

    def change_label(self):
        if self.image_viewer.selected_mask_idx >= 0:
            new_label = self.get_current_label()
            if new_label:
                self.annotations[self.image_viewer.selected_mask_idx]["label"] = new_label
                self.add_label_to_list(new_label)
                self.update_mask_list()

    def get_current_label(self):
        return self.label_input.currentText().strip()

    def add_label_to_list(self, label):
        if label and label not in self.labels:
            self.labels.append(label)
            self.label_input.addItem(label)

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
        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Sukces", "Adnotacje zapisane pomyślnie!")
            self.update_image_list()
        except Exception as e:
            logger.error(f"Błąd zapisu adnotacji {annotation_path}: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd zapisu adnotacji: {e}")

    def download_results(self):
        if not self.image_paths or not self.annotation_files:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Brak wyników do pobrania!")
            return

        # Zmień katalog zapisu na backend/data
        data_dir = os.path.join(os.getcwd(), "backend", "data")
        os.makedirs(data_dir, exist_ok=True)

        zip_path = os.path.join(data_dir, f"results_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if self.mode_combo.currentText() == "Automatyczne oznaczanie" and self.job_name:
                try:
                    logger.debug(f"Pobieram wyniki z /get_results/{self.job_name}")
                    response = requests.get(f"{self.api_url}/get_results/{self.job_name}")
                    response.raise_for_status()
                    temp_zip_path = os.path.join(data_dir, f"temp_{self.job_name}_results.zip")
                    with open(temp_zip_path, "wb") as f:
                        f.write(response.content)
                    logger.debug(f"Zapisano tymczasowy ZIP do {temp_zip_path}")

                    temp_extract_dir = os.path.join(data_dir, f"temp_extract_{self.job_name}")
                    os.makedirs(temp_extract_dir, exist_ok=True)
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                    logger.debug(f"Rozpakowano ZIP do {temp_extract_dir}")

                    self.log_directory_structure(temp_extract_dir)

                    after_dir = self.find_results_dir(temp_extract_dir, self.job_name)
                    if not after_dir:
                        logger.error(f"Katalog z wynikami auto-labelingu nie istnieje w {temp_extract_dir}!")
                        QtWidgets.QMessageBox.warning(self, "Błąd", "Katalog z wynikami auto-labelingu nie istnieje!")
                        return

                    for file_path in glob.glob(os.path.join(after_dir, "*")):
                        zipf.write(file_path, os.path.join(f"{self.job_name}_after", os.path.basename(file_path)))

                    shutil.rmtree(temp_extract_dir)
                    os.remove(temp_zip_path)
                    logger.debug(f"Usunięto tymczasowe pliki: {temp_extract_dir}, {temp_zip_path}")

                except requests.exceptions.RequestException as e:
                    logger.error("Błąd podczas pobierania wyników: %s", e)
                    QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas pobierania wyników: {e}")
                    return

            for image_path, annotation_path in zip(self.image_paths, self.annotation_files):
                zipf.write(image_path, os.path.basename(image_path))
                if os.path.exists(annotation_path):
                    zipf.write(annotation_path, os.path.basename(annotation_path))

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Zapisz wyniki", os.path.basename(zip_path), "ZIP files (*.zip)"
        )
        if save_path:
            shutil.move(zip_path, save_path)
            QtWidgets.QMessageBox.information(self, "Sukces", f"Wyniki zapisane do: {save_path}")
        else:
            os.remove(zip_path)
            logger.debug(f"Usunięto tymczasowy ZIP: {zip_path}")

    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_image_idx < len(self.image_paths) - 1:
            self.current_image_idx += 1
            self.load_current_image()