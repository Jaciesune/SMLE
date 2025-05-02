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
from PIL import Image

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, image_path=None, annotations=None, auto_labeling_tab=None):
        super().__init__(auto_labeling_tab)
        self.auto_labeling_tab = auto_labeling_tab
        self.image_path = image_path
        self.image = None
        self.qimage = None
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
                self.qimage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], 
                                         self.image.strides[0], QtGui.QImage.Format_RGB888)
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
        self.max_scale = 20.0
        self.drawing = False
        self.current_polygon = []
        self.is_panning = False
        self.last_global_pos = None
        self.mouse_pos = QtCore.QPoint(0, 0)
        self.sensitivity = 1.0  # Czułość przesuwania
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.adjust_initial_scale()

    def adjust_initial_scale(self):
        if self.image is None or not self.auto_labeling_tab.image_viewer_container:
            logger.warning("Brak obrazu lub kontenera w adjust_initial_scale")
            return

        container = self.auto_labeling_tab.image_viewer_container
        viewport_size = container.viewport().size()
        container_width = viewport_size.width()
        container_height = viewport_size.height()

        image_aspect = self.original_width / self.original_height
        container_aspect = container_width / container_height

        if image_aspect > container_aspect:
            self.scale = container_width / self.original_width
        else:
            self.scale = container_height / self.original_height

        # Ustaw min_scale tak, aby obraz nie był mniejszy niż viewport
        self.min_scale = min(container_width / self.original_width, container_height / self.original_height)
        logger.debug("Początkowa skala: %s, min_scale: %s", self.scale, self.min_scale)
        self.update_viewer_size()

    def update_viewer_size(self):
        if self.image is None:
            logger.warning("Brak obrazu w update_viewer_size")
            return

        scaled_width = int(self.original_width * self.scale)
        scaled_height = int(self.original_height * self.scale)

        self.setMinimumSize(scaled_width, scaled_height)
        container = self.auto_labeling_tab.image_viewer_container
        viewport_size = container.viewport().size()
        container_width = viewport_size.width()
        container_height = viewport_size.height()
        final_width = max(scaled_width, container_width)
        final_height = max(scaled_height, container_height)
        self.setFixedSize(final_width, final_height)

        # Ustaw zakresy pasków przewijania
        h_scroll = container.horizontalScrollBar()
        v_scroll = container.verticalScrollBar()
        h_scroll.setRange(0, max(0, scaled_width - container_width))
        v_scroll.setRange(0, max(0, scaled_height - container_height))

        logger.debug("Zaktualizowano rozmiar widżetu: %sx%s, skala: %s, h_range: %s, v_range: %s",
                     final_width, final_height, self.scale, h_scroll.maximum(), v_scroll.maximum())
        self.update()

    def wheelEvent(self, event):
        logger.debug("wheelEvent wywołane: angleDelta=%s, modifiers=%s", event.angleDelta(), QtWidgets.QApplication.keyboardModifiers())
        
        if self.image is None:
            logger.warning("Brak wczytanego obrazu, pomijam zdarzenie kółka myszy")
            return

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        container = self.auto_labeling_tab.image_viewer_container
        h_scroll = container.horizontalScrollBar()
        v_scroll = container.verticalScrollBar()

        if modifiers & QtCore.Qt.ControlModifier:
            logger.debug("Wykryto Ctrl, przetwarzam zoom")
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            mouse_pos = event.pos()

            # Sprawdź, czy obraz jest w pełni widoczny w viewport
            viewport_size = container.viewport().size()
            viewport_width = viewport_size.width()
            viewport_height = viewport_size.height()
            scaled_width = self.original_width * self.scale
            scaled_height = self.original_height * self.scale

            # Zablokuj pomniejszanie, jeśli obraz jest w pełni widoczny
            if zoom_factor < 1.0 and scaled_width <= viewport_width and scaled_height <= viewport_height:
                logger.debug("Pomniejszanie zablokowane: obraz w pełni widoczny")
                return

            # Oblicz pozycję myszy w przestrzeni obrazu
            viewport_x = mouse_pos.x() + h_scroll.value()
            viewport_y = mouse_pos.y() + v_scroll.value()
            image_x = viewport_x / self.scale
            image_y = viewport_y / self.scale

            old_scale = self.scale
            self.scale *= zoom_factor
            self.scale = max(self.min_scale, min(self.max_scale, self.scale))

            self.update_viewer_size()

            # Oblicz nowe wymiary obrazu po zmianie skali
            new_scaled_width = self.original_width * self.scale
            new_scaled_height = self.original_height * self.scale

            # Oblicz nowe wartości przewijania, aby zachować punkt pod kursorem i ograniczyć do granic obrazu
            new_viewport_x = (image_x * self.scale) - mouse_pos.x()
            new_viewport_y = (image_y * self.scale) - mouse_pos.y()

            # Ogranicz przewijanie, aby obraz nie wychodził poza granice
            max_h_scroll = max(0, new_scaled_width - viewport_width)
            max_v_scroll = max(0, new_scaled_height - viewport_height)
            new_viewport_x = max(0, min(new_viewport_x, max_h_scroll))
            new_viewport_y = max(0, min(new_viewport_y, max_v_scroll))

            # Jeśli obraz jest mniejszy niż viewport, wyśrodkuj go
            if new_scaled_width <= viewport_width:
                new_viewport_x = (viewport_width - new_scaled_width) // 2
            if new_scaled_height <= viewport_height:
                new_viewport_y = (viewport_height - new_scaled_height) // 2

            h_scroll.setValue(int(new_viewport_x))
            v_scroll.setValue(int(new_viewport_y))

            container.viewport().update()
            self.auto_labeling_tab.update_status_bar()
            self.update()
        else:
            delta = event.angleDelta().y()
            scroll_step = 50
            if modifiers & QtCore.Qt.ShiftModifier:
                new_h_value = h_scroll.value() - (delta // 120 * scroll_step)
                h_scroll.setValue(max(0, min(new_h_value, h_scroll.maximum())))
            else:
                new_v_value = v_scroll.value() - (delta // 120 * scroll_step)
                v_scroll.setValue(max(0, min(new_v_value, v_scroll.maximum())))

            container.viewport().update()
            self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.image is None or self.qimage is None:
            painter.fillRect(0, 0, self.width(), self.height(), QtGui.QColor(200, 200, 200))
            painter.drawText(self.width() // 2 - 50, self.height() // 2, "Brak obrazu")
            return

        container = self.auto_labeling_tab.image_viewer_container
        h_scroll = container.horizontalScrollBar().value()
        v_scroll = container.verticalScrollBar().value()

        scaled_image = self.qimage.scaled(
            int(self.original_width * self.scale),
            int(self.original_height * self.scale),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        painter.drawImage(-h_scroll, -v_scroll, scaled_image)

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

            scaled_mask = cv2.resize(
                full_mask,
                (int(self.original_width * self.scale), int(self.original_height * self.scale)),
                interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            color = (0, 255, 0, 128) if idx != self.selected_mask_idx else (255, 0, 0, 128)
            brush = QtGui.QBrush(QtGui.QColor(*color))
            painter.setBrush(brush)
            pen = QtGui.QPen(QtGui.QColor(*color[:3]), 2 / self.scale)
            painter.setPen(pen)
            for contour in contours:
                qpoints = [QtCore.QPoint(pt[0][0] - h_scroll, pt[0][1] - v_scroll) for pt in contour]
                polygon = QtGui.QPolygon(qpoints)
                painter.drawPolygon(polygon)

        if self.drawing and self.current_polygon:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 255), 2 / self.scale)
            painter.setPen(pen)
            scaled_points = [(pt[0] * self.scale - h_scroll, pt[1] * self.scale - v_scroll) for pt in self.current_polygon]
            qpoints = [QtCore.QPoint(int(pt[0]), int(pt[1])) for pt in scaled_points]
            for i in range(len(qpoints) - 1):
                painter.drawLine(qpoints[i], qpoints[i + 1])
            for pt in qpoints:
                painter.drawEllipse(pt, 3 / self.scale, 3 / self.scale)

    def mousePressEvent(self, event):
        if self.image is None:
            return

        if event.button() == QtCore.Qt.MidButton:
            self.is_panning = True
            self.last_global_pos = event.globalPos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return

        if event.button() == QtCore.Qt.LeftButton:
            container = self.auto_labeling_tab.image_viewer_container
            h_scroll = container.horizontalScrollBar().value()
            v_scroll = container.verticalScrollBar().value()

            x_image = (event.x() + h_scroll) / self.scale
            y_image = (event.y() + v_scroll) / self.scale

            x_image = max(0, min(self.original_width - 1, x_image))
            y_image = max(0, min(self.original_height - 1, y_image))

            if self.auto_labeling_tab.edit_mode:
                selected_idx = -1
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

                    if 0 <= int(x_image) < self.original_width and 0 <= int(y_image) < self.original_height and full_mask[int(y_image), int(x_image)] > 0:
                        selected_idx = idx
                        break

                if selected_idx >= 0:
                    self.selected_mask_idx = selected_idx
                    self.drawing = False
                    self.current_polygon = []
                    self.auto_labeling_tab.update_mask_list()
                    self.update()
            else:
                self.selected_mask_idx = -1
                if not self.drawing:
                    self.drawing = True
                    self.current_polygon = []
                if self.drawing:
                    self.current_polygon.append((x_image, y_image))
                    self.update()

    def mouseMoveEvent(self, event):
        if self.image is None:
            return

        self.mouse_pos = event.pos()
        self.auto_labeling_tab.update_status_bar()

        if self.is_panning:
            new_global_pos = event.globalPos()
            if self.last_global_pos is None:
                self.last_global_pos = new_global_pos
                return

            # Oblicz przesunięcie na podstawie globalnych współrzędnych, odwracając kierunek
            delta_x = (self.last_global_pos.x() - new_global_pos.x()) * self.sensitivity
            delta_y = (self.last_global_pos.y() - new_global_pos.y()) * self.sensitivity

            container = self.auto_labeling_tab.image_viewer_container
            h_scroll = container.horizontalScrollBar()
            v_scroll = container.verticalScrollBar()

            # Aktualizuj pozycję przewijania
            new_h_value = h_scroll.value() + int(delta_x)
            new_v_value = v_scroll.value() + int(delta_y)

            # Ogranicz wartości przewijania
            h_scroll.setValue(max(0, min(new_h_value, h_scroll.maximum())))
            v_scroll.setValue(max(0, min(new_v_value, v_scroll.maximum())))

            # Zaktualizuj pozycję globalną
            self.last_global_pos = new_global_pos

            # Odśwież widok
            self.update()
            return

        if self.drawing:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MidButton:
            self.is_panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.last_global_pos = None

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

            x_min = max(0, min(self.original_width - 1, int(x_min)))
            x_max = max(0, min(self.original_width - 1, int(x_max)))
            y_min = max(0, min(self.original_height - 1, int(y_min)))
            y_max = max(0, min(self.original_height - 1, int(y_max)))

            if x_max <= x_min or y_max <= y_min:
                QtWidgets.QMessageBox.warning(self, "Błąd", "Nieprawidłowe granice maski!")
                self.drawing = False
                self.current_polygon = []
                self.update()
                return

            mask = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
            points = np.array([(int(pt[0]), int(pt[1])) for pt in self.current_polygon], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            cropped_mask = mask[y_min:y_max, x_min:x_max]
            if cropped_mask.size == 0:
                QtWidgets.QMessageBox.warning(self, "Błąd", "Maska jest pusta!")
                self.drawing = False
                self.current_polygon = []
                self.update()
                return

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
                self.auto_labeling_tab.add_to_history(new_shape)
                self.auto_labeling_tab.update_mask_list()
                self.auto_labeling_tab.add_label_to_list(label)
                self.auto_labeling_tab.save_annotations(silent=True)

            self.drawing = False
            self.current_polygon = []
            self.update()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete and self.selected_mask_idx >= 0:
            self.auto_labeling_tab.delete_mask()
        elif event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_S:
            self.auto_labeling_tab.save_annotations()
        elif event.key() == QtCore.Qt.Key_Escape and self.drawing:
            self.drawing = False
            self.current_polygon = []
            self.update()
        elif event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_Z:
            self.auto_labeling_tab.undo_mask()
        elif event.key() == QtCore.Qt.Key_Left:
            self.auto_labeling_tab.prev_image()
        elif event.key() == QtCore.Qt.Key_Right:
            self.auto_labeling_tab.next_image()
        elif event.key() == QtCore.Qt.Key_E:
            self.auto_labeling_tab.toggle_edit_mode()

    def get_mouse_position(self):
        if self.image is None:
            return 0, 0
        container = self.auto_labeling_tab.image_viewer_container
        h_scroll = container.horizontalScrollBar().value()
        v_scroll = container.verticalScrollBar().value()
        x_image = (self.mouse_pos.x() + h_scroll) / self.scale
        y_image = (self.mouse_pos.y() + v_scroll) / self.scale
        return int(x_image), int(y_image)

class AutoLabelingTab(QtWidgets.QWidget):
    def __init__(self, user_role):
        super().__init__()
        self.user_role = user_role
        self.api_url = "http://localhost:8000"
        self.job_name = None
        self.temp_dir = None
        self.input_dir = None
        self.original_image_paths = []
        self.image_paths = []
        self.annotation_files = []
        self.current_image_idx = 0
        self.annotations = []
        self.labels = ["obiekt"]
        self.default_label = "obiekt"  # Domyślna etykieta dla wszystkich obrazów
        self.history = []
        self.edit_mode = False
        self.init_ui()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout()

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.image_viewer_container = QtWidgets.QScrollArea()
        self.image_viewer_container.setWidgetResizable(False)
        self.image_viewer = ImageViewer(auto_labeling_tab=self)
        self.image_viewer_container.setWidget(self.image_viewer)
        self.image_viewer_container.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.image_viewer_container.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.main_splitter.addWidget(self.image_viewer_container)

        self.right_panel = QtWidgets.QWidget()
        self.right_panel.setFixedWidth(400)
        self.right_layout = QtWidgets.QVBoxLayout()

        self.right_panel.setStyleSheet("""
            QLabel {
                font-size: 14px;
            }
            QComboBox, QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 13px;
            }
            QComboBox:hover, QLineEdit:hover {
                border: 1px solid #888;
            }
            QPushButton {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
                font-size: 13px;
            }
        """)

        self.mode_widget = QtWidgets.QWidget()
        mode_layout = QtWidgets.QVBoxLayout()

        mode_row = QtWidgets.QHBoxLayout()
        self.mode_label = QtWidgets.QLabel("Tryb:")
        self.mode_label.setFixedWidth(80)
        mode_row.addWidget(self.mode_label)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Ręczne oznaczanie", "Automatyczne oznaczanie"])
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)

        label_row = QtWidgets.QHBoxLayout()
        self.custom_label_label = QtWidgets.QLabel("Etykieta:")
        self.custom_label_label.setFixedWidth(80)
        label_row.addWidget(self.custom_label_label)
        self.custom_label_input = QtWidgets.QLineEdit()
        self.custom_label_input.setPlaceholderText("np. plank, pipe")
        label_row.addWidget(self.custom_label_input)
        label_row.addStretch()
        mode_layout.addLayout(label_row)

        model_row = QtWidgets.QHBoxLayout()
        self.model_version_label = QtWidgets.QLabel("Model:")
        self.model_version_label.setFixedWidth(80)
        model_row.addWidget(self.model_version_label)
        self.model_version_combo = QtWidgets.QComboBox()
        model_row.addWidget(self.model_version_combo)
        model_row.addStretch()
        mode_layout.addLayout(model_row)

        self.mode_widget.setLayout(mode_layout)
        self.right_layout.addWidget(self.mode_widget)
        self.right_layout.addSpacing(10)

        self.toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Wczytaj")
        self.load_btn.setIcon(QtGui.QIcon.fromTheme("document-open"))
        self.load_btn.clicked.connect(self.load_and_label)
        toolbar_layout.addWidget(self.load_btn)
        self.save_btn = QtWidgets.QPushButton("Zapisz")
        self.save_btn.setIcon(QtGui.QIcon.fromTheme("document-save"))
        self.save_btn.clicked.connect(self.save_annotations)
        toolbar_layout.addWidget(self.save_btn)
        self.undo_btn = QtWidgets.QPushButton("Cofnij")
        self.undo_btn.setIcon(QtGui.QIcon.fromTheme("edit-undo"))
        self.undo_btn.clicked.connect(self.undo_mask)
        self.undo_btn.setEnabled(False)
        toolbar_layout.addWidget(self.undo_btn)
        self.shortcuts_btn = QtWidgets.QPushButton("Skróty")
        self.shortcuts_btn.setIcon(QtGui.QIcon.fromTheme("help-about"))
        self.shortcuts_btn.clicked.connect(self.show_shortcuts)
        toolbar_layout.addWidget(self.shortcuts_btn)
        toolbar_layout.addStretch()
        self.toolbar.setLayout(toolbar_layout)
        self.right_layout.addWidget(self.toolbar)
        self.right_layout.addSpacing(15)

        self.label_widget = QtWidgets.QWidget()
        self.label_layout = QtWidgets.QHBoxLayout()
        self.label_input_label = QtWidgets.QLabel("Etykieta:")
        self.label_input_label.setStyleSheet("font-weight: bold;")
        self.label_layout.addWidget(self.label_input_label)
        self.label_input = QtWidgets.QComboBox()
        self.label_input.setEditable(True)
        self.label_input.addItems(self.labels)
        self.label_layout.addWidget(self.label_input)
        self.label_widget.setLayout(self.label_layout)
        self.right_layout.addWidget(self.label_widget)
        self.right_layout.addSpacing(15)

        self.mask_list_widget = QtWidgets.QWidget()
        self.mask_list_layout = QtWidgets.QVBoxLayout()
        self.mask_list_label = QtWidgets.QLabel("Lista masek:")
        self.mask_list_label.setStyleSheet("font-weight: bold;")
        self.mask_list_layout.addWidget(self.mask_list_label)
        self.mask_list = QtWidgets.QListWidget()
        self.mask_list.itemClicked.connect(self.select_mask)
        self.mask_list_layout.addWidget(self.mask_list)
        self.delete_mask_btn = QtWidgets.QPushButton("Usuń (Del)")
        self.delete_mask_btn.clicked.connect(self.delete_mask)
        self.mask_list_layout.addWidget(self.delete_mask_btn)
        self.change_label_btn = QtWidgets.QPushButton("Zmień etykietę")
        self.change_label_btn.clicked.connect(self.change_label)
        self.mask_list_layout.addWidget(self.change_label_btn)
        self.download_btn = QtWidgets.QPushButton("Pobierz wyniki")
        self.download_btn.clicked.connect(self.download_results)
        self.download_btn.setEnabled(False)
        self.mask_list_layout.addWidget(self.download_btn)
        self.mask_list_widget.setLayout(self.mask_list_layout)
        self.right_layout.addWidget(self.mask_list_widget)
        self.right_layout.addSpacing(15)

        self.image_list_widget = QtWidgets.QWidget()
        self.image_list_layout = QtWidgets.QVBoxLayout()
        self.image_list_label = QtWidgets.QLabel("Lista obrazów:")
        self.image_list_label.setStyleSheet("font-weight: bold;")
        self.image_list_layout.addWidget(self.image_list_label)
        self.image_list = QtWidgets.QListWidget()
        self.image_list.itemClicked.connect(self.select_image)
        self.image_list_layout.addWidget(self.image_list)
        self.image_navigation = QtWidgets.QHBoxLayout()
        self.prev_image_btn = QtWidgets.QPushButton("Poprzedni")
        self.prev_image_btn.clicked.connect(self.prev_image)
        self.next_image_btn = QtWidgets.QPushButton("Następny")
        self.next_image_btn.clicked.connect(self.next_image)
        self.image_navigation.addWidget(self.prev_image_btn)
        self.image_navigation.addWidget(self.next_image_btn)
        self.image_list_layout.addLayout(self.image_navigation)
        self.image_list_widget.setLayout(self.image_list_layout)
        self.right_layout.addWidget(self.image_list_widget)

        self.right_panel.setLayout(self.right_layout)
        self.main_splitter.addWidget(self.right_panel)

        self.main_splitter.setSizes([int(self.width() * 0.75), int(self.width() * 0.25)])
        main_layout.addWidget(self.main_splitter)

        self.status_bar = QtWidgets.QLabel("Zoom: 100% | Pozycja: (0, 0) | Tryb: Oznaczanie")
        main_layout.addWidget(self.status_bar)

        self.setLayout(main_layout)
        self.update_model_versions()

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        mode_text = "Edycja" if self.edit_mode else "Oznaczanie"
        self.status_bar.setText(f"Zoom: {int(self.image_viewer.scale * 100 / self.image_viewer.min_scale)}% | Pozycja: {self.image_viewer.get_mouse_position()} | Tryb: {mode_text}")
        self.image_viewer.selected_mask_idx = -1
        self.update_mask_list()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'image_viewer') and self.image_viewer:
            self.image_viewer.adjust_initial_scale()

    def update_status_bar(self):
        if not hasattr(self, 'image_viewer') or self.image_viewer.image is None:
            self.status_bar.setText("Zoom: 100% | Pozycja: (0, 0) | Tryb: Oznaczanie")
            return
        zoom_percent = int(self.image_viewer.scale * 100 / self.image_viewer.min_scale)
        x, y = self.image_viewer.get_mouse_position()
        mode_text = "Edycja" if self.edit_mode else "Oznaczanie"
        self.status_bar.setText(f"Zoom: {zoom_percent}% | Pozycja: ({x}, {y}) | Tryb: {mode_text}")

    def select_input_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Wybierz katalog ze zdjęciami")
        return directory

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
        input_dir = self.select_input_directory()
        if not input_dir:
            return

        self.input_dir = input_dir
        mode = self.mode_combo.currentText()

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

            QtWidgets.QMessageBox.information(self, "Informacja", "Rozpoczynam automatyczne oznaczanie...")

            progress_dialog = QtWidgets.QProgressDialog("Oznaczanie w toku...", "", 0, 0, self)
            progress_dialog.setWindowTitle("Przetwarzanie")
            progress_dialog.setCancelButton(None)
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setRange(0, 0)
            QtWidgets.QApplication.processEvents()

            self.job_name = f"auto_label_{uuid.uuid4().hex}"
            try:
                files = [('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in self.original_image_paths]
                data = {
                    'job_name': self.job_name,
                    'model_version': model_version,
                    'custom_label': custom_label
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
                    progress_dialog.close()
                    QtWidgets.QMessageBox.warning(self, "Błąd", result.get('message', 'Nieznany błąd'))
                    return

                if "message" in result and "Nie znaleziono obiektów" in result["message"]:
                    progress_dialog.close()
                    QtWidgets.QMessageBox.information(self, "Informacja", "Auto-labeling nie znalazł żadnych obiektów. Możesz przejść do ręcznego oznaczania.")
                    self.current_image_idx = 0
                    self.load_current_image()
                    self.update_image_list()
                    self.download_btn.setEnabled(True)
                    return

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
                    progress_dialog.close()
                    QtWidgets.QMessageBox.warning(self, "Błąd", "Katalog z wynikami auto-labelingu nie istnieje!")
                    return

                self.image_paths = glob.glob(os.path.join(after_dir, "*.jpg"))
                self.annotation_files = glob.glob(os.path.join(after_dir, "*.json"))
                self.image_paths.sort()
                self.annotation_files.sort()

                if not self.image_paths or not self.annotation_files:
                    progress_dialog.close()
                    QtWidgets.QMessageBox.warning(self, "Błąd", "Brak wyników auto-labelingu (obrazów lub adnotacji)!")
                    return

                if len(self.image_paths) != len(self.annotation_files):
                    progress_dialog.close()
                    QtWidgets.QMessageBox.warning(self, "Błąd", "Liczba obrazów i adnotacji nie jest zgodna!")
                    return

            except requests.exceptions.RequestException as e:
                progress_dialog.close()
                QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd podczas labelowania: {e}")
                return
            finally:
                for _, file_tuple in files:
                    file_tuple[1].close()
                progress_dialog.close()

        self.current_image_idx = 0
        self.load_current_image()
        self.update_image_list()
        self.download_btn.setEnabled(True)

    def load_current_image(self):
        if not self.image_paths:
            logger.error("Brak ścieżek do obrazów!")
            return

        if self.current_image_idx >= len(self.image_paths):
            logger.error(f"Nieprawidłowy indeks obrazu: {self.current_image_idx}")
            return

        image_path = self.image_paths[self.current_image_idx]
        annotation_path = self.annotation_files[self.current_image_idx] if self.current_image_idx < len(self.annotation_files) else None

        if not os.path.exists(image_path):
            logger.error(f"Plik obrazu nie istnieje: {image_path}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Plik obrazu nie istnieje: {image_path}")
            return

        self.annotations = []
        if annotation_path and os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.annotations = json_data["shapes"]
                # Znajdź pierwszą etykietę na obrazie i ustaw jako domyślną
                if self.annotations:
                    first_label = self.annotations[0]["label"]
                    self.default_label = first_label
                    logger.debug(f"Ustawiono domyślną etykietę dla obrazu {image_path}: {first_label}")
                    # Ustaw wszystkie maski na tę samą etykietę
                    for shape in self.annotations:
                        shape["label"] = first_label
            except Exception as e:
                logger.error(f"Błąd wczytywania adnotacji {annotation_path}: {e}")
                QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd wczytywania adnotacji: {e}")

        if hasattr(self, 'image_viewer') and self.image_viewer is not None:
            self.image_viewer = None

        try:
            self.image_viewer = ImageViewer(image_path, self.annotations, auto_labeling_tab=self)
            self.image_viewer_container.setWidget(self.image_viewer)
            self.image_viewer.setFocus(QtCore.Qt.OtherFocusReason)
            self.image_viewer.adjust_initial_scale()
        except Exception as e:
            logger.error(f"Błąd wczytywania obrazu {image_path}: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd wczytywania obrazu: {e}")
            return

        # Ustaw domyślną etykietę w polu wyboru
        custom_label = self.custom_label_input.text().strip()
        if custom_label:
            self.default_label = custom_label
            self.label_input.setCurrentText(custom_label)
            if custom_label not in self.labels:
                self.labels.append(custom_label)
                self.label_input.addItem(custom_label)
        else:
            self.label_input.setCurrentText(self.default_label)
            if self.default_label not in self.labels:
                self.labels.append(self.default_label)
                self.label_input.addItem(self.default_label)

        self.update_mask_list()
        self.image_list.setCurrentRow(self.current_image_idx)
        self.update_status_bar()

    def update_mask_list(self):
        self.mask_list.clear()
        for idx, shape in enumerate(self.annotations):
            self.mask_list.addItem(f"Maska {idx + 1}: {shape['label']}")
        if self.image_viewer.selected_mask_idx >= 0:
            self.mask_list.setCurrentRow(self.image_viewer.selected_mask_idx)

    def update_image_list(self):
        self.image_list.clear()
        for i, image_path in enumerate(self.image_paths):
            item = QtWidgets.QListWidgetItem(os.path.basename(image_path))
            annotation_path = self.annotation_files[i] if i < len(self.annotation_files) else None
            has_annotations = False
            if annotation_path and os.path.exists(annotation_path):
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
        if self.edit_mode:
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
            self.save_annotations(silent=True)

    def add_to_history(self, shape):
        self.history.append(shape)
        self.undo_btn.setEnabled(True)

    def undo_mask(self):
        if self.history:
            last_shape = self.history.pop()
            if last_shape in self.annotations:
                self.annotations.remove(last_shape)
            self.image_viewer.selected_mask_idx = -1
            self.update_mask_list()
            self.image_viewer.update()
            self.save_annotations(silent=True)
        if not self.history:
            self.undo_btn.setEnabled(False)

    def change_label(self):
        if self.image_viewer.selected_mask_idx >= 0:
            new_label = self.get_current_label()
            if new_label:
                # Zmiana etykiety na wszystkich maskach na obrazie
                for shape in self.annotations:
                    shape["label"] = new_label
                self.default_label = new_label
                self.add_label_to_list(new_label)
                self.update_mask_list()
                self.save_annotations(silent=True)

    def get_current_label(self):
        label = self.label_input.currentText().strip()
        if not label:
            label = self.default_label
        return label

    def add_label_to_list(self, label):
        if label and label not in self.labels:
            self.labels.append(label)
            self.label_input.addItem(label)

    def save_annotations(self, silent=False):
        if not self.image_paths:
            logger.warning("Brak ścieżek do obrazów, pomijam zapis adnotacji")
            return

        if self.mode_combo.currentText() == "Ręczne oznaczanie":
            if not self.input_dir:
                logger.error("Brak input_dir w trybie ręcznym")
                QtWidgets.QMessageBox.warning(self, "Błąd", "Katalog wejściowy nie został wybrany!")
                return
            annotation_dir = self.input_dir
        else:
            annotation_dir = os.path.dirname(self.image_paths[self.current_image_idx])

        annotation_path = os.path.join(
            annotation_dir,
            f"{os.path.splitext(os.path.basename(self.image_paths[self.current_image_idx]))[0]}.json"
        )

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
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Adnotacje zapisane do: {annotation_path}")
            if not silent:
                QtWidgets.QMessageBox.information(self, "Sukces", "Adnotacje zapisane pomyślnie!")
            self.update_image_list()
        except PermissionError as e:
            logger.error(f"Brak uprawnień do zapisu pliku {annotation_path}: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Brak uprawnień do zapisu pliku: {e}")
        except Exception as e:
            logger.error(f"Błąd zapisu adnotacji {annotation_path}: {e}")
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Błąd zapisu adnotacji: {e}")

    def show_shortcuts(self):
        shortcuts_text = (
            "Lista skrótów klawiaturowych:\n\n"
            "Strzałka w lewo: Poprzedni obraz\n"
            "Strzałka w prawo: Następny obraz\n"
            "Ctrl + S: Zapisz adnotacje\n"
            "Ctrl + Z: Cofnij dodanie maski\n"
            "Delete: Usuń wybraną maskę\n"
            "Esc: Anuluj rysowanie maski\n"
            "E: Przełącz tryb (Oznaczanie/Edycja)\n"
            "Ctrl + kółko myszy: Przybliż/oddal obraz\n"
            "Kółko myszy: Przewiń obraz w pionie\n"
            "Shift + kółko myszy: Przewiń obraz w poziomie\n"
            "Środkowy przycisk myszy: Przesuwaj zdjęcie"
        )
        QtWidgets.QMessageBox.information(self, "Skróty klawiaturowe", shortcuts_text)

    def download_results(self):
        if not self.image_paths:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Brak wyników do pobrania!")
            return

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

            for image_path in self.image_paths:
                zipf.write(image_path, os.path.basename(image_path))
                annotation_path = os.path.join(os.path.dirname(image_path), f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
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