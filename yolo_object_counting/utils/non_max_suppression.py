import torch

def debug_boxes(box1, box2):
    print("\n🧐 DEBUG - Sample Box1 & Box2 before IoU computation:")
    print(f"Box1:\n{box1}")
    print(f"Box2:\n{box2}")
    print(f"Box1 min: {box1.min()}, max: {box1.max()}")
    print(f"Box2 min: {box2.min()}, max: {box2.max()}")

def box_iou(box1, box2):
    """Oblicza IoU dla bboxów w formacie (x1, y1, x2, y2)"""
    if box2.shape[0] == 0:
        return torch.zeros((1, box1.shape[0]))

    # 🔍 Debug: sprawdź, czy boxy mają poprawne wartości
    debug_boxes(box1, box2)

    # Oblicz przecięcie bboxów
    inter_x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Oblicz pola bboxów
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area

    iou = inter_area / (union_area + 1e-6)

    # 🛠️ Debug: Sprawdź podejrzane IoU
    if (iou < 0).any():
        print(f"⚠️ BŁĄD! Znaleziono ujemne wartości IoU:\n{iou}")

    return iou



def non_max_suppression(predictions, conf_threshold=0.7, iou_threshold=0.5):
    output = []
    
    for pred in predictions:
        if pred is None or not len(pred):
            output.append(None)
            continue

        print(f"Before filtering - predictions shape: {pred.shape}")  
        print(f"Sample predictions before filtering:\n{pred[:5]}")

        scores = pred[:, 4]  # Confidence score
        mask = scores > conf_threshold  # Filtrujemy predykcje poniżej progu
        pred = pred[mask]

        print(f"After filtering - remaining predictions: {len(pred)}")  
        if not len(pred):
            output.append(None)
            continue

        # Sortowanie po score
        _, indices = scores[mask].sort(descending=True)
        pred = pred[indices]

        print(f"Initial pred shape after sorting: {pred.shape}")
        
        keep = []
        iteration = 0
        while pred.size(0):
            iteration += 1
            keep.append(pred[0].unsqueeze(0))
            if len(pred) == 1:
                break

            box1 = pred[0, :4].unsqueeze(0)
            box2 = pred[1:, :4]
            print(f"Iteration {iteration} - Box1: {box1.shape}, Box2: {box2.shape}")

            try:
                ious = box_iou(box1, box2)
                print(f"Iteration {iteration} - IoUs:\n{ious}")

                filtered_pred = pred[1:][ious[0] < iou_threshold]
                pred = filtered_pred
                print(f"Iteration {iteration} - Updated pred shape: {pred.shape}")

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                print(f"pred shape: {pred.shape}, box1 shape: {box1.shape}, box2 shape: {box2.shape}")
                raise

        final_detections = torch.cat(keep) if len(keep) else None
        print(f"Before NMS: {predictions[0].shape[0]} boxes, After NMS: {len(final_detections) if final_detections is not None else 0} boxes")

        output.append(final_detections)

    return output
