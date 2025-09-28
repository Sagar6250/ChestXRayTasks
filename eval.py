import torch
from settings import *

def dice_coefficient(preds, targets, eps=1e-6):
    # preds, targets are tensors [B,1,H,W] with 0/1 values
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)
    return dice.item()

def evaluate(model, dataloader, task_type="classification"):
    model.eval()
    correct = 0
    total_mae = 0
    n_samples =0
    dice_scores = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device=DEVICE)
            labels = labels.to(device=DEVICE)
            outputs = model(images)
            # print(outputs)
            if task_type == 'classification':
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(f"Test Accuracy: {100*correct/total:.2f}%") 

            if task_type == 'regression':
                outputs = outputs.squeeze()
                total_mae += torch.abs(outputs - labels).sum().item()
                n_samples += labels.size(0)       
                avg_mae = total_mae / n_samples
                # print(f"Test MAE: {avg_mae:.4f}")

            if task_type == 'segmentation':
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                dice = dice_coefficient(preds, labels)
                dice_scores.append(dice)