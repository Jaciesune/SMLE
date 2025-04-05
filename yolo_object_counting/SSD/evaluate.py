import torch
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer  # Poprawna nazwa klasy
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="SSD Evaluation")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint")
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    model = SSDDetector(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)  # Poprawna nazwa klasy
    checkpoint = checkpointer.load(args.checkpoint)
    if not checkpoint:
        raise ValueError(f"Could not load checkpoint from {args.checkpoint}")

    print(f"Loaded checkpoint from {args.checkpoint} at iteration {checkpoint.get('iteration', 0)}")
    eval_results = do_evaluation(cfg, model, distributed=False, iteration=checkpoint.get("iteration", 0))
    for dataset_name, result in eval_results.items():
        print(f"Evaluation on {dataset_name}: {result}")


if __name__ == "__main__":
    main()