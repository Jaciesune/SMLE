import os
import torch
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.transforms = transforms

        # Filtrujemy tylko obrazy z poprawnymi adnotacjami
        self.ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if not anns:
                continue
            has_valid_box = any(ann["bbox"][2] > 1 and ann["bbox"][3] > 1 for ann in anns)
            if has_valid_box:
                self.ids.append(img_id)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 1 and h > 1:  # pomijamy zerowe/ujemne boxy
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
        return img, target, path

    def __len__(self):
        return len(self.ids)