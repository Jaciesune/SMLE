from PyQt5 import QtWidgets, QtCore

class ArchiveTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Główny layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Tabela dla Archiwum
        self.archive_table = QtWidgets.QTableWidget()
        self.archive_table.setRowCount(1)  # Dodajemy 1 wiersz (przykładowy wpis)
        self.archive_table.setColumnCount(5)  # 5 kolumn: Model, Algorytm, Operacja, Data, Użytkownik

        # Ustawiamy nagłówki kolumn
        self.archive_table.setHorizontalHeaderLabels(["Model", "Algorytm", "Operacja", "Data", "Użytkownik"])

        # Wypełniamy tabelę przykładowymi danymi
        self.archive_table.setItem(0, 0, QtWidgets.QTableWidgetItem("Model Testowy CNN"))
        self.archive_table.setItem(0, 1, QtWidgets.QTableWidgetItem("CNN"))
        self.archive_table.setItem(0, 2, QtWidgets.QTableWidgetItem("Zliczanie"))
        self.archive_table.setItem(0, 3, QtWidgets.QTableWidgetItem("08:13 08.03.2025"))
        self.archive_table.setItem(0, 4, QtWidgets.QTableWidgetItem("Uzytkownik_TEST"))

        # Zablokowanie edytowania danych
        self.archive_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

        # Pobieramy nagłówek tabeli
        header = self.archive_table.horizontalHeader()

        # Zablokowanie zmiany rozmiaru kolumn
        header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        # Ustawienie stałej szerokości dla każdej kolumny
        self.archive_table.setColumnWidth(0, 200)
        self.archive_table.setColumnWidth(1, 200)
        self.archive_table.setColumnWidth(2, 200)
        self.archive_table.setColumnWidth(3, 200)
        self.archive_table.setColumnWidth(4, 200)

        # Włączamy rozciąganie tabeli, aby zajmowała całą przestrzeń
        self.archive_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Dodajemy tabelę do layoutu
        layout.addWidget(self.archive_table, stretch=1)

        # Układ przycisków
        btn_layout = QtWidgets.QHBoxLayout()
        self.view_details_btn = QtWidgets.QPushButton("Zobacz szczegóły")
        btn_layout.addWidget(self.view_details_btn)

        # Dodajemy przycisk do layoutu
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def view_details(self):
        # Funkcja przenosząca do szczegółów (implementacja zależna od wymagań)
        QtWidgets.QMessageBox.information(self, "Szczegóły", "Wyświetlanie szczegółów operacji...")
