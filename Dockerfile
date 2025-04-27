# Używamy obrazu bazowego z obsługą CUDA 12.8.0
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Ustawienie katalogu roboczego
WORKDIR /app

# Instalacja Pythona 3.10 i pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Instalacja zależności systemowych dla OpenCV, PyQt5, Qt i innych bibliotek
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libxcb1 \
    libx11-xcb1 \
    qt5-qmake \
    qtbase5-dev \
    qtchooser \
    qt5-qmake-bin \
    qtbase5-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie pliku requirements.txt
COPY backend/requirements.txt ./requirements.txt

# Instalacja zależności Pythona z requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie całego kodu projektu do kontenera
COPY . /app

# Ustawienie zmiennych środowiskowych dla Qt (offscreen rendering, jeśli używane)
ENV QT_QPA_PLATFORM=offscreen
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Domyślne polecenie uruchamiające backend
CMD ["python", "/app/backend/main.py"]