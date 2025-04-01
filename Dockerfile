# Używamy obrazu bazowego z CUDA i Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Ustawienie katalogu roboczego
WORKDIR /app

# Aktualizacja i instalacja podstawowych narzędzi
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ustawienie Python 3.10 jako domyślnego
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Uaktualnienie pip
RUN python3 -m pip install --upgrade pip

# Instalacja zależności systemowych dla OpenCV i PyQt5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libxcb1 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie zależności backendu
COPY backend/requirements.txt ./backend/requirements.txt

# Instalowanie zależności backendu
RUN pip install --no-cache-dir -r backend/requirements.txt

# Instalacja PyTorch z obsługą GPU (dla CUDA 12.2) oraz innych zależności
RUN pip install --no-cache-dir --pre \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    && pip install --no-cache-dir \
    opencv-python-headless==4.11.0.86 \
    pillow==11.0.0 \
    numpy==2.1.2 \
    matplotlib==3.9.2

# Kopiowanie tylko potrzebnych plików backendu
COPY backend/ /app/backend/

# Ustawienie zmiennej środowiskowej dla Qt, aby działało w trybie offscreen
ENV QT_QPA_PLATFORM=offscreen
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Uruchomienie backendu
CMD ["python", "/app/backend/main.py"]