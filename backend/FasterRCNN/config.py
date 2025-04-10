# Detekcja
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3  # sensowna wartość

# Augmentacja i obraz
INPUT_SIZE = 1024

# Filtry dla bounding boxów – teraz z zakresem min i max
MAX_ASPECT_RATIO = 1.5
MIN_ASPECT_RATIO = 0.7
MAX_BOX_AREA_RATIO = 0.3
MIN_BOX_AREA_RATIO = 0.001

# Model selection (dla checkpointów)
SAVE_PERFECT_MODEL_RATIO_RANGE = (0.93, 1.07)

# Anchory (dopasowane do obiektów widzianych z przodu – np. rury, deski)
ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
ANCHOR_RATIOS = (
    (0.8, 1.0, 1.2),
    (0.8, 1.0, 1.2),
    (0.8, 1.0, 1.2),
    (0.8, 1.0, 1.2),
    (0.8, 1.0, 1.2)
)

# Liczba klas
NUM_CLASSES = 2  # background + klasa_rury

# Włącz/Wyłącz anchory
USE_CUSTOM_ANCHORS = False
