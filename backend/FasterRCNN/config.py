# Detekcja
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3  # <- wcześniej było 15000, co nie miało sensu :)

# Augmentacja i obraz
INPUT_SIZE = 1024

# Filtry dla bounding boxów
MAX_ASPECT_RATIO = 3.5
MAX_BOX_AREA_RATIO = 0.3

# Model selection (dla checkpointów)
SAVE_PERFECT_MODEL_RATIO_RANGE = (0.93, 1.07)

# Anchory (dla rur - smukłe i cienkie)
ANCHOR_SIZES = ((32, 64, 128, 256, 512),)
ANCHOR_RATIOS = ((0.1, 0.2, 0.5, 1.0, 2.0, 5.0),)

# Liczba klas
NUM_CLASSES = 2  # background + klasa_rury

# Włącz/Wyłącz anchory
USE_CUSTOM_ANCHORS = True
