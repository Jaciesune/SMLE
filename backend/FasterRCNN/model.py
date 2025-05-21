"""
Moduł modelu Faster R-CNN

Ten moduł dostarcza funkcje do inicjalizacji modelu detekcji obiektów Faster R-CNN,
z obsługą konfigurowalnych parametrów, niestandardowych kotwic (anchors) i optymalizacji
pod kątem detekcji małych obiektów. Umożliwia elastyczny wybór między różnymi wersjami
modeli bazowych i automatyczną walidację kompatybilności.
"""

#######################
# Importy bibliotek
#######################
import torch                                     # Framework PyTorch do treningu modeli głębokich sieci neuronowych
import torchvision                               # Modele, transformacje i zbiory danych do wizji komputerowej
from torchvision.models.detection import FasterRCNN  # Implementacja modelu Faster R-CNN
from torchvision.models.detection.rpn import AnchorGenerator  # Generator kotwic dla RPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # Klasyfikator dla roi_heads
import logging                                   # Do logowania informacji i błędów
from config import ANCHOR_SIZES, ANCHOR_RATIOS, USE_CUSTOM_ANCHORS, NUM_CLASSES, NMS_THRESHOLD, USE_FASTER_RCNN_V2

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_model(num_classes, device):
    """
    Tworzy i konfiguruje model Faster R-CNN.
    
    Inicjalizuje model Faster R-CNN z siecią ResNet50 i Feature Pyramid Network (FPN),
    stosując pretrenowane wagi z domyślnych zestawów danych torchvision. Umożliwia 
    konfigurację niestandardowych kotwic (anchors), progów detekcji i innych parametrów
    dostosowanych do wykrywania małych obiektów.
    
    Args:
        num_classes (int): Liczba klas (włącznie z tłem).
        device (torch.device): Urządzenie, na którym będzie działać model (CPU/GPU).
        
    Returns:
        torchvision.models.detection.FasterRCNN: Skonfigurowany model Faster R-CNN.
        
    Raises:
        ValueError: Gdy parametry kotwic są niepoprawne.
    """
    #######################
    # Sprawdzanie wersji bibliotek
    #######################
    # Logowanie wersji PyTorch i torchvision
    logger.info(f"Używana wersja PyTorch: {torch.__version__}")
    logger.info(f"Używana wersja torchvision: {torchvision.__version__}")

    # Walidacja wersji
    required_pytorch_version = "2.7"
    required_torchvision_version = "0.22"
    if not torch.__version__.startswith(required_pytorch_version):
        logger.warning(f"Zalecana wersja PyTorch to {required_pytorch_version}.x, używana: {torch.__version__}")
    if not torchvision.__version__.startswith(required_torchvision_version):
        logger.warning(f"Zalecana wersja torchvision to {required_torchvision_version}.x, używana: {torchvision.__version__}")

    #######################
    # Konfiguracja kotwic (anchors)
    #######################
    # Walidacja parametrów anchorów
    if USE_CUSTOM_ANCHORS:
        if not ANCHOR_SIZES or not ANCHOR_RATIOS:
            raise ValueError("ANCHOR_SIZES i ANCHOR_RATIOS muszą być zdefiniowane w config.py, gdy USE_CUSTOM_ANCHORS=True")
        anchor_generator = AnchorGenerator(
            sizes=ANCHOR_SIZES,
            aspect_ratios=ANCHOR_RATIOS
        )
        logger.info(f"Używam niestandardowych anchorów: sizes={ANCHOR_SIZES}, aspect_ratios={ANCHOR_RATIOS}")
    else:
        anchor_generator = None
        logger.info("Używam domyślnych anchorów modelu Faster R-CNN")

    #######################
    # Parametry RPN (Region Proposal Network)
    #######################
    # Definicja parametrów RPN
    rpn_pre_nms_top_n_train = 2000
    rpn_post_nms_top_n_train = 1000
    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_test = 300

    #######################
    # Inicjalizacja modelu
    #######################
    # Wybór wersji modelu
    model_func = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2 if USE_FASTER_RCNN_V2 else torchvision.models.detection.fasterrcnn_resnet50_fpn
    logger.info(f"Używam modelu: {'fasterrcnn_resnet50_fpn_v2' if USE_FASTER_RCNN_V2 else 'fasterrcnn_resnet50_fpn'}")

    # Inicjalizacja modelu
    try:
        model = model_func(
            weights='DEFAULT',  # Pretrenowane wagi
            box_nms_thresh=max(0.1, min(NMS_THRESHOLD, 0.7)),  # Rozsądny próg NMS
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_batch_size_per_image=256,
            box_detections_per_img=500,  # Zmniejszono dla mniejszego zużycia pamięci
            box_score_thresh=0.2  # Obniżony próg dla większej liczby predykcji
        )
    except TypeError as e:
        # Obsługa błędów kompatybilności dla różnych wersji torchvision
        logger.error(f"Błąd inicjalizacji modelu: {str(e)}")
        if "rpn_anchor_generator" in str(e):
            logger.info("Próba inicjalizacji bez jawnego rpn_anchor_generator")
            model = model_func(
                weights='DEFAULT',
                box_nms_thresh=max(0.1, min(NMS_THRESHOLD, 0.7)),
                rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                rpn_batch_size_per_image=256,
                box_detections_per_img=500,
                box_score_thresh=0.2
            )
            if USE_CUSTOM_ANCHORS:
                logger.info("Nadpisywanie domyślnego generatora anchorów niestandardowym")
                model.rpn.anchor_generator = anchor_generator

    #######################
    # Dostosowanie klasyfikatora
    #######################
    # Modyfikacja klasyfikatora dla liczby klas
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Przeniesienie modelu na urządzenie
    model.to(device)

    #######################
    # Logowanie konfiguracji modelu
    #######################
    # Logowanie parametrów modelu
    logger.info(f"Model Faster R-CNN działa na: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"Parametry modelu: num_classes={num_classes}, "
                f"box_score_thresh={model.roi_heads.score_thresh}, "
                f"box_nms_thresh={model.roi_heads.nms_thresh}, "
                f"box_detections_per_img={model.roi_heads.detections_per_img}, "
                f"rpn_pre_nms_top_n_train={rpn_pre_nms_top_n_train}, "
                f"rpn_post_nms_top_n_train={rpn_post_nms_top_n_train}, "
                f"rpn_pre_nms_top_n_test={rpn_pre_nms_top_n_test}, "
                f"rpn_post_nms_top_n_test={rpn_post_nms_top_n_test}")

    return model