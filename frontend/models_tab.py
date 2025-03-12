from PyQt5 import QtWidgets

class ModelsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.models_list = QtWidgets.QListWidget()
        self.models_list.addItem("Model 1")
        self.models_list.addItem("Model 2")
        layout.addWidget(self.models_list)

        btn_layout = QtWidgets.QHBoxLayout()
        self.load_model_btn = QtWidgets.QPushButton("Wczytaj model")
        self.delete_model_btn = QtWidgets.QPushButton("Usuń model")
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.delete_model_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
