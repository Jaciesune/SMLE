from PyQt5 import QtWidgets, QtGui, QtCore

class CountTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Główny układ poziomy
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # Usuwamy marginesy głównego układu, aby kontener był wycentrowany

        # Tworzymy kontener dla obu stron, wycentrowany na ekranie
        container_widget = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout()

        # Układ dla lewej strony - zdjęcie
        left_layout = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel("Tutaj pojawi się zdjęcie")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(500, 400)  # Ustawienie rozmiaru zdjęcia
        self.image_label.setStyleSheet("border: 1px solid #606060; background-color: #767676;")
        left_layout.addWidget(self.image_label)

        # Tworzymy kontener dla prawej strony
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setFixedHeight(400)  # Ustawiamy wysokość prawej strony na 400px
        right_widget.setFixedWidth(750) # Ustawiamy szerokość prawej strony na 750px
        
        # Dodajemy lewą i prawą część do głównego układu
        container_layout.addLayout(left_layout)
        container_layout.addWidget(right_widget)

        # Ustawiamy kontener na środku ekranu
        container_widget.setLayout(container_layout)
        main_layout.addWidget(container_widget)

        # Ustawiamy główny układ
        self.setLayout(main_layout)
        right_widget.setLayout(right_layout) # Ustawiamy główny układ w kontenerze prawej strony
        right_layout.setAlignment(QtCore.Qt.AlignTop) # Ustawienie wyrównania elementów w prawym układzie do góry

        # Przycisk "Wczytaj zdjęcie"
        self.load_btn = QtWidgets.QPushButton("Wczytaj zdjęcie")
        self.load_btn.clicked.connect(self.load_image)
        right_layout.addWidget(self.load_btn)

        # Lista rozwijana "Wybierz Algorytm"
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItem("Wybierz Algorytm")
        self.algorithm_combo.setItemData(0, True, QtCore.Qt.ItemIsEnabled)  # Ustawiamy jako niewybieralny
        self.algorithm_combo.addItem("CNN")
        self.algorithm_combo.addItem("R-CNN")
        self.algorithm_combo.addItem("Mask R-CNN")
        self.algorithm_combo.setCurrentIndex(0)  # Ustawienie na domyślny element
        right_layout.addWidget(self.algorithm_combo)

        # Przycisk "Rozpocznij analizę"
        self.analyze_btn = QtWidgets.QPushButton("Rozpocznij analizę")
        self.analyze_btn.clicked.connect(self.analyze_image)
        right_layout.addWidget(self.analyze_btn)


    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz zdjęcie", "", "Obrazy (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            pixmap = QtGui.QPixmap(file_path).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

    def analyze_image(self):
        # Pobieramy wybrany algorytm z listy rozwijanej
        selected_algorithm = self.algorithm_combo.currentText()
        QtWidgets.QMessageBox.information(self, "Analiza", f"Algorytm wybrany: {selected_algorithm}\nAnaliza obrazu została uruchomiona.")
