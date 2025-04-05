import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.data.transforms import build_transforms, build_target_transform
from ssd.structures.container import Container


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids


def make_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0):
    # Wybór transformacji: z augmentacją dla treningu, bez dla walidacji
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    dataset = build_dataset(
        dataset_list,
        transform=train_transform,
        target_transform=target_transform,
        is_train=is_train
    )  # Pojedynczy dataset

    batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
    num_workers = cfg.DATA_LOADER.NUM_WORKERS
    pin_memory = cfg.DATA_LOADER.PIN_MEMORY
    shuffle = is_train and not distributed

    sampler = make_batch_data_sampler(
        dataset,
        batch_size,
        max_iter if is_train else None,
        start_iter,
        distributed=distributed,
    )
    collator = BatchCollator(is_train=is_train)
    data_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )
    
    print(f"DEBUG: Number of data loaders created: 1")
    print(f"DEBUG: Created dataset '{dataset_list[0]}' with {len(dataset)} items")
    return data_loader


def make_batch_data_sampler(dataset, batch_size, num_iters=None, start_iter=0, distributed=False):
    if distributed:
        sampler = samplers.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset) if num_iters else torch.utils.data.SequentialSampler(dataset)
    batch_sampler = samplers.IterationBasedBatchSampler(sampler, batch_size, num_iters, start_iter)
    return batch_sampler