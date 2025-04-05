import torch
from ssd.data.build import make_data_loader
from ssd.utils.checkpoint import CheckPointer
import os
import time
from tqdm import tqdm
from ssd.data.datasets.evaluation import evaluate
import numpy as np


@torch.no_grad()
def do_evaluation(cfg, model, distributed=False, iteration=None):
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    output_folder = cfg.OUTPUT_DIR

    data_loader_val = make_data_loader(
        cfg,
        is_train=False,
        distributed=distributed,
    )

    eval_results = []
    dataset_name = cfg.DATASETS.TEST[0]  # "val"
    print(f"Evaluating dataset: {dataset_name}")
    
    eval_result = inference(
        model,
        data_loader_val,
        dataset_name,
        device,
        output_folder,
        iteration=iteration,
    )
    
    # Wyświetlanie liczby wykrytych obiektów dla każdego obrazu
    if 'detections_per_image' in eval_result:
        print(f"\nLiczba wykrytych obiektów na zdjęciu w zbiorze {dataset_name}:")
        for detection in eval_result['detections_per_image']:
            print(f"Image ID: {detection['image_id']}, Wykryto obiektów: {detection['num_detections']}")
    
    # Przygotowanie metryk do zwrócenia
    metrics = {
        'precision': eval_result.get('precision', 0.0),
        'recall': eval_result.get('recall', 0.0),
        'mAP': eval_result.get('mAP', 0.0),  # jeśli jest
        'AP50': eval_result.get('AP50', 0.0),  # itd.
    }

    eval_results.append({'metrics': metrics})

    model.train()
    return eval_results


def inference(model, data_loader, dataset_name, device, output_folder, **kwargs):
    model.eval()
    predictions = []
    detections_per_image = []  # Lista do przechowywania liczby wykrytych obiektów na zdjęciu
    timer = time.time()
    num_images = len(data_loader.dataset)
    print(f"Processing {num_images} images from {dataset_name}")

    # Przetwarzaj batche i mapuj predykcje na indeksy obrazów
    img_id_to_pred = {}
    for i, (images, _, img_ids) in enumerate(tqdm(data_loader, unit="batch")):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        for img_id, output in zip(img_ids, outputs):
            # Przetwarzanie predykcji dla każdego obrazu
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            # Filtrowanie predykcji na podstawie progu ufności
            confidence_threshold = 0.01  # Zgodne z cfg.TEST.CONFIDENCE_THRESHOLD
            mask = scores > confidence_threshold
            num_detections = mask.sum()  # Liczba wykrytych obiektów po filtracji

            # Zapis predykcji
            img_id_to_pred[img_id.item()] = {
                "boxes": boxes[mask],
                "labels": labels[mask],
                "scores": scores[mask],
            }

            # Zapis liczby wykrytych obiektów
            detections_per_image.append({
                'image_id': img_id.item(),  # Używamy img_id jako identyfikator
                'num_detections': num_detections
            })

    # Upewnij się, że predictions ma dokładnie num_images elementów
    predictions = [img_id_to_pred.get(i, {"boxes": np.array([]), "labels": np.array([]), "scores": np.array([])})
                   for i in range(num_images)]

    print(f"Inference took {time.time() - timer:.3f} seconds")
    
    # Obliczanie metryk za pomocą evaluate
    result = evaluate(
        dataset=data_loader.dataset,
        predictions=predictions,
        output_dir=output_folder,
        **kwargs,
    )
    
    # Dodanie informacji o liczbie wykrytych obiektów do wyniku
    result['detections_per_image'] = detections_per_image
    
    return result