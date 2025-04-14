from PyQt5 import QtWidgets
import requests

class ModelsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_models()  # Załaduj modele przy starcie

    def init_ui(self):
        # Główny layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tworzymy tabelę do wyświetlania modeli
        self.models_table = QtWidgets.QTableWidget()
        self.models_table.setRowCount(0)  # Zaczynamy od pustej tabeli
        self.models_table.setColumnCount(8)  # Zmieniamy na 8 kolumn, aby dodać 'training_date'

        # Ustawiamy nagłówki kolumn
        self.models_table.setHorizontalHeaderLabels([
            "ID", "Nazwa Modelu", "Algorytm", "Wersja", "Dokładność", 
            "Data Utworzenia", "Status", "Data Treningu"
        ])

        # Zablokowanie edytowania danych
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Zablokowanie zmiany rozmiaru kolumn
        self.models_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.models_table.setColumnWidth(0, 50)  # Szerokość pierwszej kolumny
        self.models_table.setColumnWidth(1, 250)  # Szerokość drugiej kolumny
        self.models_table.setColumnWidth(2, 150)  # Szerokość trzeciej kolumny
        self.models_table.setColumnWidth(3, 100)  # Szerokość czwartej kolumny
        self.models_table.setColumnWidth(4, 100)  # Szerokość piątej kolumny
        self.models_table.setColumnWidth(5, 150)  # Szerokość szóstej kolumny
        self.models_table.setColumnWidth(6, 100)  # Szerokość siódmej kolumny
        self.models_table.setColumnWidth(7, 150)  # Szerokość ósmej kolumny

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

    def load_models(self):
        """Pobiera modele z API backendu i wyświetla w tabeli"""
        try:
            response = requests.get("http://localhost:8000/models")  # Połączenie z backendem
            response.raise_for_status()  # Sprawdzenie, czy odpowiedź jest poprawna

            models = response.json()
            self.display_models(models)  # 🔥 Wywołujemy funkcję wyświetlania danych
        except requests.exceptions.RequestException as e:
            QtWidgets.QMessageBox.warning(self, "Błąd", f"Nie udało się pobrać danych: {e}")

    def display_models(self, models):
        """Wypełnia tabelę modelami"""
        self.models_table.setRowCount(len(models))
        for row, model in enumerate(models):
            # Jeżeli nie ma 'training_date', ustawiamy pustą komórkę
            training_date = model.get("training_date", "")  # Jeżeli brak, przypisujemy pustą wartość

            self.models_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(model["id"])))
            self.models_table.setItem(row, 1, QtWidgets.QTableWidgetItem(model["name"]))
            self.models_table.setItem(row, 2, QtWidgets.QTableWidgetItem(model["algorithm"]))
            self.models_table.setItem(row, 3, QtWidgets.QTableWidgetItem(model["version"]))
            self.models_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(model["accuracy"])))
            self.models_table.setItem(row, 5, QtWidgets.QTableWidgetItem(model["creation_date"]))
            self.models_table.setItem(row, 6, QtWidgets.QTableWidgetItem(model["status"]))
            self.models_table.setItem(row, 7, QtWidgets.QTableWidgetItem(training_date))  # Ustawiamy pustą komórkę, jeżeli brak danych
       
    def create_new_model(self):
        # Funkcja przenosząca do zakładki TrainTab
        print("Przenoszenie do TrainTab...")
