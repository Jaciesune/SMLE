import torch


def reduce_loss_dict(loss_dict):
    """Redukuje słownik strat z wielu GPU do pojedynczej wartości."""
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        with torch.no_grad():
            keys = list(loss_dict.keys())
            vals = [loss_dict[k].mean() for k in keys]
            torch.distributed.all_reduce(torch.stack(vals))
            return {k: v.item() for k, v in zip(keys, vals)}
    return {k: v.item() for k, v in loss_dict.items()}