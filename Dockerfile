# Bazowy obraz Pythona 3.10
FROM python:3.10

# Ustawienie katalogu roboczego
WORKDIR /app

# Instalacja zależności systemowych dla PyQt5 i Qt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxcb1 \
    libx11-xcb1 \
    qt5-qmake \
    qtbase5-dev \
    qtchooser \
    qt5-qmake-bin \
    qtbase5-dev-tools

# Kopiowanie zależności backendu
COPY backend/requirements.txt ./backend/requirements.txt
COPY frontend/requirements.txt ./frontend/requirements.txt

# Instalowanie zależności backendu
RUN pip install --no-cache-dir -r backend/requirements.txt

# Instalowanie zależności frontendu
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Kopiowanie całego kodu (backend i frontend) do kontenera
COPY . /app

# Ustawienie zmiennej środowiskowej dla Qt, aby działało w trybie offscreen
ENV QT_QPA_PLATFORM=offscreen

ENV XDG_RUNTIME_DIR=/tmp/runtime-root


# Uruchomienie backendu i frontendu jednocześnie
CMD bash -c "python /app/backend/main.py & python /app/frontend/main.py"


