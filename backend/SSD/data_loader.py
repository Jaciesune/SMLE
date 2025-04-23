import os
import torch
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        boxes = torch.tensor([[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in boxes], dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
        return img, target, path

    def __len__(self):
        return len(self.ids)