"""
Implementacja architektury Multi-Column Convolutional Neural Network (MCNN)
dla problemu zliczania obiektów i generowania map gęstości.

MCNN wykorzystuje równoległe ścieżki konwolucyjne z filtrami o różnych rozmiarach
do jednoczesnego wykrywania cech obiektów w różnych skalach, co czyni model
szczególnie efektywnym w zliczaniu obiektów o zróżnicowanych rozmiarach.

Autorzy oryginalnej pracy: Zhang et al., "Single-Image Crowd Counting via Multi-Column CNN"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN(nn.Module):
    """
    Multi-Column Convolutional Neural Network do zliczania obiektów.
    
    Architektura składa się z trzech równoległych ścieżek konwolucyjnych,
    z których każda wykorzystuje filtry o różnych rozmiarach:
    - branch1: duże filtry (9x9, 7x7) do wykrywania dużych obiektów
    - branch2: średnie filtry (7x7, 5x5) do obiektów średniej wielkości
    - branch3: małe filtry (5x5, 3x3) do wykrywania mniejszych obiektów
    
    Wszystkie ścieżki są następnie łączone i przetwarzane przez warstwę fuzji,
    która generuje jednowymiarową mapę gęstości jako wyjście.
    """
    def __init__(self):
        """
        Inicjalizacja modelu MCNN z trzema równoległymi ścieżkami konwolucyjnymi
        o różnych rozmiarach filtrów.
        """
        super(MCNN, self).__init__()

        # 🔹 Ścieżka 1: Duże filtry - do wykrywania dużych obiektów/wzorców
        self.branch1 = nn.Sequential(
            # Warstwa wejściowa: konwersja 3-kanałowego obrazu na 64 mapy cech z filtrem 9x9
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),  # padding=4 zachowuje wymiary przestrzenne
            nn.ReLU(),  # Nieliniowa aktywacja ReLU
            # Druga warstwa: redukcja do 32 map cech z filtrem 7x7
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        # 🔹 Ścieżka 2: Średnie filtry - do wykrywania obiektów średniej wielkości
        self.branch2 = nn.Sequential(
            # Warstwa wejściowa: konwersja 3-kanałowego obrazu na 64 mapy cech z filtrem 7x7
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            # Druga warstwa: redukcja do 32 map cech z filtrem 5x5
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        # 🔹 Ścieżka 3: Małe filtry - do wykrywania małych obiektów/szczegółów
        self.branch3 = nn.Sequential(
            # Warstwa wejściowa: konwersja 3-kanałowego obrazu na 64 mapy cech z filtrem 5x5
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # Druga warstwa: redukcja do 32 map cech z filtrem 3x3
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 🔹 Warstwa fuzji - łączy wyniki wszystkich ścieżek
        self.fuse = nn.Sequential(
            # Redukcja połączonych 96 map cech (32*3) do 64 map z konwolucją 1x1
            nn.Conv2d(96, 64, kernel_size=1),  # konwolucja 1x1 łączy informacje z różnych kanałów
            nn.ReLU(),
            # Finalna konwersja do pojedynczej mapy gęstości
            nn.Conv2d(64, 1, kernel_size=1)  # Wyjście: mapa gęstości o wymiarach wejścia
        )

    def forward(self, x):
        """
        Przeprowadza forward pass przez sieć MCNN.
        
        Args:
            x (torch.Tensor): Tensor wejściowy o kształcie [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Mapa gęstości o kształcie [batch_size, 1, height, width]
        """
        # Przetwarzanie obrazu przez trzy równoległe ścieżki
        b1 = self.branch1(x)  # Ekstrakcja cech z dużymi filtrami
        b2 = self.branch2(x)  # Ekstrakcja cech z średnimi filtrami
        b3 = self.branch3(x)  # Ekstrakcja cech z małymi filtrami

        # Połączenie wyników z trzech ścieżek wzdłuż wymiaru kanałów (dim=1)
        # Wynikowy tensor ma wymiar [batch_size, 96, height, width]
        out = torch.cat((b1, b2, b3), dim=1)
        
        # Zastosowanie warstwy fuzji do połączonych map cech
        # Wynikowa mapa gęstości ma wymiar [batch_size, 1, height, width]
        out = self.fuse(out)
        
        return out