import torch

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    print(f"\nEpoka {epoch}... (Batchy: {len(dataloader)})")

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, new_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Strata: {loss.item():.4f}")

    return total_loss / len(dataloader)
