# config.py
# Detekcja
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5  # Standardowa wartość

# Augmentacja i obraz
INPUT_SIZE = 1024

# Filtry dla bounding boxów
MIN_ASPECT_RATIO = 0.01
MAX_ASPECT_RATIO = 9999
MIN_BOX_AREA_RATIO = 0.00001
MAX_BOX_AREA_RATIO = 1.0

# Model selection
SAVE_PERFECT_MODEL_RATIO_RANGE = (0.93, 1.07)
USE_FASTER_RCNN_V2 = True  # Użyj fasterrcnn_resnet50_fpn_v2

# Anchory
ANCHOR_SIZES = ((8, 16, 32, 64, 128),)  # Mniejsze rozmiary dla małych obiektów
ANCHOR_RATIOS = ((0.2, 0.5, 1.0, 2.0, 5.0),)  # Szerszy zakres proporcji
USE_CUSTOM_ANCHORS = True  # Włącz dla małych obiektów

# Liczba klas
NUM_CLASSES = 2

# Optymalizator i scheduler
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-6