import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
import torch
from torchvision.transforms import functional as F


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(image_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(self.image_dir, path)).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)


def get_data_loaders(train_path, val_path, train_annotations, val_annotations, batch_size=2, num_workers=2):
    transforms = T.Compose([T.ToTensor()])

    train_dataset = CocoDataset(train_path, train_annotations, transforms=transforms)
    val_dataset = CocoDataset(val_path, val_annotations, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader
