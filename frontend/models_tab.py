from PyQt5 import QtWidgets, QtCore

class ModelsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Główny layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Nagłówki kolumn: Nazwa Modelu, Algorytm, Data Utworzenia
        self.models_table = QtWidgets.QTableWidget()
        self.models_table.setRowCount(2)  # Dodajemy 2 wiersze (przykładowe modele)
        self.models_table.setColumnCount(3)  # 3 kolumny: Nazwa Modelu, Algorytm, Data Utworzenia

        # Ustawiamy nagłówki kolumn
        self.models_table.setHorizontalHeaderLabels(["Nazwa Modelu", "Algorytm", "Data Utworzenia"])

        # Wypełniamy tabelę przykładowymi danymi
        self.models_table.setItem(0, 0, QtWidgets.QTableWidgetItem("Model Testowy CNN"))
        self.models_table.setItem(0, 1, QtWidgets.QTableWidgetItem("CNN"))
        self.models_table.setItem(0, 2, QtWidgets.QTableWidgetItem("10.03.2025"))

        self.models_table.setItem(1, 0, QtWidgets.QTableWidgetItem("Model Testowy R-CNN"))
        self.models_table.setItem(1, 1, QtWidgets.QTableWidgetItem("R-CNN"))
        self.models_table.setItem(1, 2, QtWidgets.QTableWidgetItem("11.03.2025"))

        # Zablokowanie edytowania danych
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Zablokowanie zmiany rozmiaru kolumn
        self.models_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.models_table.setColumnWidth(0, 300)  # Szerokość pierwszej kolumny
        self.models_table.setColumnWidth(1, 300)  # Szerokość drugiej kolumny
        self.models_table.setColumnWidth(2, 200)  # Szerokość trzeciej kolumny

        # Dodajemy tabelę do layoutu
        layout.addWidget(self.models_table)

        # Układ przycisków
        btn_layout = QtWidgets.QHBoxLayout()
        self.load_model_btn = QtWidgets.QPushButton("Wczytaj model")
        self.delete_model_btn = QtWidgets.QPushButton("Usuń model")
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.delete_model_btn)

        # Przycisk "Utwórz Nowy Model"
        self.create_new_model_btn = QtWidgets.QPushButton("Utwórz Nowy Model")
        self.create_new_model_btn.clicked.connect(self.create_new_model)

        # Dodajemy przycisk do layoutu
        btn_layout.addWidget(self.create_new_model_btn)
        
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def create_new_model(self):
        # Funkcja przenosząca do zakładki TrainTab
        print("Przenoszenie do TrainTab...")
        
