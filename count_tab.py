from PyQt5 import QtWidgets, QtGui, QtCore

class CountTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.load_btn = QtWidgets.QPushButton("Wczytaj zdjęcie")
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        self.image_label = QtWidgets.QLabel("Tutaj pojawi się zdjęcie")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(500, 350)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #fff;")
        layout.addWidget(self.image_label)

        self.analyze_btn = QtWidgets.QPushButton("Rozpocznij analizę")
        self.analyze_btn.clicked.connect(self.analyze_image)
        layout.addWidget(self.analyze_btn)

        self.setLayout(layout)

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wybierz zdjęcie", "", "Obrazy (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            pixmap = QtGui.QPixmap(file_path).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

    def analyze_image(self):
        QtWidgets.QMessageBox.information(self, "Analiza", "Analiza obrazu została uruchomiona.")
