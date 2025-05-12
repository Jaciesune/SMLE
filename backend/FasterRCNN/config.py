"""
Plik konfiguracyjny dla modelu detekcji obiektów Faster R-CNN

Ten plik zawiera parametry konfiguracyjne dla modelu detekcji obiektów
opartego na architekturze Faster R-CNN, zoptymalizowanego pod kątem
wykrywania małych obiektów.
"""

#######################
# Parametry detekcji
#######################
CONFIDENCE_THRESHOLD = 0.4  # Minimalny próg pewności dla uznania detekcji za poprawną
NMS_THRESHOLD = 0.5         # Próg dla Non-Maximum Suppression (usuwanie nakładających się ramek)

#######################
# Parametry obrazu
#######################
INPUT_SIZE = 1024           # Rozmiar wejściowy obrazu dla modelu

#######################
# Filtry dla ramek (bounding boxes)
#######################
MIN_ASPECT_RATIO = 0.01     # Minimalny stosunek szerokości do wysokości ramki
MAX_ASPECT_RATIO = 9999     # Maksymalny stosunek szerokości do wysokości ramki
MIN_BOX_AREA_RATIO = 0.00001  # Minimalny stosunek powierzchni ramki do powierzchni obrazu
MAX_BOX_AREA_RATIO = 1.0    # Maksymalny stosunek powierzchni ramki do powierzchni obrazu

#######################
# Wybór modelu
#######################
SAVE_PERFECT_MODEL_RATIO_RANGE = (0.93, 1.07)  # Zakres współczynnika oceny uznawany za "idealny"
USE_FASTER_RCNN_V2 = True   # Używaj nowszej wersji modelu (fasterrcnn_resnet50_fpn_v2)

#######################
# Konfiguracja kotwic (anchors)
#######################
# Kotwice to prostokątne ramki o różnych rozmiarach, używane jako szablony do wykrywania obiektów
ANCHOR_SIZES = ((8, 16, 32, 64, 128),)  # Mniejsze rozmiary kotwic dla małych obiektów
ANCHOR_RATIOS = ((0.2, 0.5, 1.0, 2.0, 5.0),)  # Szerszy zakres proporcji kotwic
USE_CUSTOM_ANCHORS = True   # Włącz używanie niestandardowych kotwic (zamiast domyślnych)

#######################
# Parametry klasyfikacji
#######################
NUM_CLASSES = 2             # Liczba klas (1 klasa obiektu + 1 klasa tła)

#######################
# Parametry optymalizacji
#######################
MOMENTUM = 0.9              # Parametr momentum dla optymalizatora SGD
WEIGHT_DECAY = 0.0005       # Parametr regularyzacji L2 (zapobiega przeuczeniu)
SCHEDULER_FACTOR = 0.5      # Współczynnik zmniejszenia współczynnika uczenia
SCHEDULER_PATIENCE = 2      # Liczba epok bez poprawy przed zmianą współczynnika uczenia
MIN_LR = 1e-6               # Minimalna wartość współczynnika uczenia