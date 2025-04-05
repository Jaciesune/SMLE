import os


class DatasetCatalog:
    _REGISTERED = {}
    DATA_DIR = 'datasets'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
    }

    @staticmethod
    def register(name, func):
        """Rejestruje nowy dataset w katalogu."""
        DatasetCatalog._REGISTERED[name] = func

    @staticmethod
    def get(name):
        """Pobiera zarejestrowany dataset."""
        if name not in DatasetCatalog._REGISTERED:
            raise RuntimeError("Dataset not available: {}".format(name))
        return DatasetCatalog._REGISTERED[name]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
