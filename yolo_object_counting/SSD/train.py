import argparse
import logging
import os
from typing import Dict, Any

import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool

# Configuration constants
CONFIG = {
    'DEFAULT_CONFIG_FILE': "configs/vgg_ssd300_pipes.yaml",  # Bez zmian, ale upewnij się, że YAML jest aktualny
    'LOG_STEP': 25,  # Zwiększam do 50, mniej spamu w logach
    'SAVE_STEP': 200,  # Zmniejszam do 200, częstsze zapisy dla małego zbioru
    'EVAL_STEP': 100,  # Zmniejszam do 200, częstsza ewaluacja
    'USE_TENSORBOARD': True,  # Bez zmian, ale dodaj metryki w kodzie
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',  # Bez zmian
    'OUTPUT_DIR': './outputs/pipes',  # Zsynchronizowane z YAML
    'CHECKPOINT_DIR': './checkpoints/pipes'  # Zsynchronizowane i bardziej specyficzne
}

def train(cfg, args) -> torch.nn.Module:
    """Train the SSD model with given configuration and arguments."""
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    lr = cfg.SOLVER.LR * args.num_gpus
    optimizer = make_optimizer(cfg, model, lr)
    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments: Dict[str, Any] = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    mkdir(CONFIG['CHECKPOINT_DIR'])
    checkpointer = CheckPointer(
        model,
        optimizer,
        scheduler,
        CONFIG['CHECKPOINT_DIR'],
        save_to_disk,
        logger
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(
        cfg,
        is_train=True,
        distributed=args.distributed,
        max_iter=max_iter,
        start_iter=arguments['iteration']
    )

    model = do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        arguments,
        args
    )
    return model

def setup_training() -> tuple:
    """Setup training environment and arguments."""
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument("--config-file", default=CONFIG['DEFAULT_CONFIG_FILE'], help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=CONFIG['LOG_STEP'], type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=CONFIG['SAVE_STEP'], type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=CONFIG['EVAL_STEP'], type=int, help='Evaluate dataset every eval_step')
    parser.add_argument('--use_tensorboard', default=CONFIG['USE_TENSORBOARD'], type=str2bool)
    parser.add_argument("--skip-test", action="store_true", help="Do not test the final model")
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.local_rank)

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    return args, num_gpus

def main():
    args, num_gpus = setup_training()

    # Wczytanie konfiguracji - przekazujemy ścieżkę, a nie otwarty plik
    cfg.merge_from_file(args.config_file)  # Poprawione
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = CONFIG['DEVICE']
    cfg.OUTPUT_DIR = CONFIG['OUTPUT_DIR']
    cfg.freeze()

    mkdir(cfg.OUTPUT_DIR)
    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(f"Arguments: {args}")
    logger.info(f"Loaded configuration file: {args.config_file}")
    
    # Logowanie zawartości pliku YAML
    with open(args.config_file, "r", encoding='utf-8') as cf:
        logger.info("\n" + cf.read())
    logger.info(f"Running with config:\n{cfg}")

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Starting evaluation...')
        torch.cuda.empty_cache()
        do_evaluation(cfg, model, distributed=args.distributed)

if __name__ == '__main__':
    main()