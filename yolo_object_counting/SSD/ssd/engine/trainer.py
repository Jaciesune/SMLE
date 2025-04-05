import datetime
import logging
import os
import time
import torch
import torch.distributed as dist
import time
import datetime
import logging
import torch
from torch.cuda.amp import GradScaler, autocast  # Dla mixed precision
from ssd.utils import dist_util, reduce_loss_dict
from collections.abc import Mapping

from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, Mapping):  # Zmień na Mapping
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(cfg, model, data_loader, optimizer, scheduler, checkpointer, device, arguments, args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = cfg.SOLVER.MAX_ITER
    batch_size = cfg.SOLVER.BATCH_SIZE
    num_images = len(data_loader.dataset) if hasattr(data_loader.dataset, '__len__') else 68
    iters_per_epoch = max(1, num_images // batch_size)
    max_epochs = max_iter // iters_per_epoch
    logger.info(f"Training for {max_epochs} epochs, {iters_per_epoch} iterations per epoch")

    use_amp = torch.cuda.is_available() and True
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    start_iter = arguments["iteration"]
    current_epoch = start_iter // iters_per_epoch
    start_training_time = time.time()
    end = time.time()

    from ssd.structures.container import Container  # Import Container

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        if iteration >= max_iter:
            break
        iteration += 1
        arguments["iteration"] = iteration

        new_epoch = iteration // iters_per_epoch
        if new_epoch > current_epoch:
            current_epoch = new_epoch
            logger.info(f"Starting epoch {current_epoch + 1}/{max_epochs}")

        # Obsługa targets jako Container
        if isinstance(targets, Container):
            targets = targets.to(device)  # Przenosimy cały Container na device
        else:
            logger.error(f"Expected targets to be a Container, got {type(targets)}")
            raise ValueError("Invalid targets format")

        images = images.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets=targets)
                loss = sum(loss for loss in loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets=targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)

        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_msg = meters.delimiter.join([
                "iter: {iter:06d}",
                "epoch: {epoch}/{max_epoch}",
                "lr: {lr:.5f}",
                '{meters}',
                "eta: {eta}",
            ]).format(
                iter=iteration,
                epoch=current_epoch + 1,
                max_epoch=max_epochs,
                lr=optimizer.param_groups[0]['lr'],
                meters=str(meters),
                eta=eta_string,
            )
            if device == "cuda":
                log_msg += f", mem: {round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)}M"
            logger.info(log_msg)

            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar(f'losses/{loss_name}', loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            checkpointer.save(f"model_{iteration:06d}", **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and iteration != max_iter:
            eval_results = do_evaluation(cfg, model, distributed=args.distributed, iteration=iteration)
            logger.info(f"Eval results: {eval_results}")  # Debugowanie
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    if isinstance(eval_result, dict) and 'metrics' in eval_result:
                        write_metric(eval_result['metrics'], f'metrics/{dataset}', summary_writer, iteration)
                    else:
                        logger.error(f"Invalid eval_result format: {eval_result}")
                        continue
            model.train()

    checkpointer.save("model_final", **arguments)
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f"Total training time: {total_time_str} ({total_training_time / max_iter:.4f} s / it)")
    return model