from .pipes import PipesDataset, register_pipes_datasets
from ssd.config.path_catlog import DatasetCatalog

register_pipes_datasets()

def build_dataset(dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) == 1, f"Expected single dataset, got: {dataset_list}"
    dataset_name = dataset_list[0]  # "train" lub "val"
    dataset_fn = DatasetCatalog.get(dataset_name)
    dataset = dataset_fn(transform=transform, target_transform=target_transform)
    print(f"DEBUG: Created dataset '{dataset_name}' with {len(dataset)} items")
    return dataset  # Zwracaj pojedynczy dataset, nie listę