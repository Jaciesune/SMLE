from PyQt5 import QtWidgets
import requests

class ModelsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_models()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.models_table = QtWidgets.QTableWidget()
        self.models_table.setRowCount(0)
        self.models_table.setColumnCount(8)
        self.models_table.setHorizontalHeaderLabels([
            "ID", "Nazwa Modelu", "Algorytm", "Wersja", "Dokładność",
            "Data Utworzenia", "Status", "Data Treningu"
        ])
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.models_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.models_table.setColumnWidth(0, 50)
        self.models_table.setColumnWidth(1, 250)
        self.models_table.setColumnWidth(2, 150)
        self.models_table.setColumnWidth(3, 100)
        self.models_table.setColumnWidth(4, 100)
        self.models_table.setColumnWidth(5, 150)
        self.models_table.setColumnWidth(6, 100)
        self.models_table.setColumnWidth(7, 150)

        layout.addWidget(self.models_table)

        btn_layout = QtWidgets.QHBoxLayout()
        self.load_model_btn = QtWidgets.QPushButton("Wczytaj model")
        self.delete_model_btn = QtWidgets.QPushButton("Usuń model")
        self.delete_model_btn.clicked.connect(self.delete_selected_model)  # ← dodano
        btn_layout.addWidget(self.load_model_btn)
        btn_layout.addWidget(self.delete_model_btn)

        self.create_new_model_btn = QtWidgets.QPushButton("Utwórz Nowy Model")
        self.create_new_model_btn.clicked.connect(self.create_new_model)
        btn_layout.addWidget(self.create_new_model_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_models(self):
        try:
            response = requests.get("http://localhost:8000/models")
            response.raise_for_status()
            models = response.json()
            self.display_models(models)
        except requests.exceptions.RequestException as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")

    def display_models(self, models):
        self.models_table.setRowCount(len(models))
        for row, model in enumerate(models):
            training_date = model.get("training_date", "")
            self.models_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(model["id"])))
            self.models_table.setItem(row, 1, QtWidgets.QTableWidgetItem(model["name"]))
            self.models_table.setItem(row, 2, QtWidgets.QTableWidgetItem(model["algorithm"]))
            self.models_table.setItem(row, 3, QtWidgets.QTableWidgetItem(model["version"]))
            self.models_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(model["accuracy"])))
            self.models_table.setItem(row, 5, QtWidgets.QTableWidgetItem(model["creation_date"]))
            self.models_table.setItem(row, 6, QtWidgets.QTableWidgetItem(model["status"]))
            self.models_table.setItem(row, 7, QtWidgets.QTableWidgetItem(training_date))

    def create_new_model(self):
        print("Przenoszenie do TrainTab...")

    def delete_selected_model(self):
        selected_row = self.models_table.currentRow()
        if selected_row < 0:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie wybrano żadnego modelu do usunięcia.")
            return

        model_id_item = self.models_table.item(selected_row, 0)
        if not model_id_item:
            QtWidgets.QMessageBox.warning(self, "Błąd", "Nie można odczytać ID modelu.")
            return

        model_id = model_id_item.text()

        confirm = QtWidgets.QMessageBox.question(
            self,
            "Potwierdzenie",
            f"Czy na pewno chcesz usunąć model o ID {model_id}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if confirm == QtWidgets.QMessageBox.Yes:
            try:
                response = requests.delete(f"http://localhost:8000/models/{model_id}")
                response.raise_for_status()
                QtWidgets.QMessageBox.information(self, "Sukces", "Model został usunięty.")
                self.load_models()
            except requests.exceptions.RequestException as e:
                QtWidgets.QMessageBox.critical(self, "Błąd", f"Nie udało się usunąć modelu: {e}")
