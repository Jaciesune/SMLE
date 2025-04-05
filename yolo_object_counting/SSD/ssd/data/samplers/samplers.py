import torch
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, num_iterations, start_iter=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_iterations = num_iterations if num_iterations is not None else len(sampler) // batch_size
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    iteration += 1
                    if iteration >= self.num_iterations:
                        break
            if batch:  # Obsługuje resztę danych
                yield batch
                iteration += 1

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(dataset, batch_size, num_iters=None, start_iter=0, distributed=False):
    """Tworzy sampler dla DataLoadera."""
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
    else:
        if num_iters is not None:  # Dla treningu
            sampler = RandomSampler(dataset)
        else:  # Dla testu/ewaluacji
            sampler = SequentialSampler(dataset)
    
    batch_sampler = IterationBasedBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        num_iterations=num_iters,
        start_iter=start_iter
    )
    return batch_sampler