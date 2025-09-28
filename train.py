from settings import *
from tqdm import tqdm
# from eval import *
from datetime import datetime
import torch
import csv

def train(epoch, model, dataloader, criterion, optimizer, task_type="classification"):
    model.train()

    for epoch in tqdm(range(epoch, epoch+args.epoch), desc= "Epoch"):
        epoch_loss = 0

        batch_bar = tqdm(dataloader, desc=f"Batches", leave=False)
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device=DEVICE), labels.to(device=DEVICE)
            optimizer.zero_grad()
            output = model(images)
            if task_type == "regression":
                output = output.squeeze()

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_bar.update(1)
            batch_bar.set_postfix(loss=loss.item())

        batch_bar.close()
        epoch_loss = epoch_loss/len(dataloader)
        tqdm.write(f"Epoch {epoch+1} train loss: {epoch_loss}")

        # if(run_epoch%2 == 0):
        #     val_loss = evaluate(model, val_loader, optimizer)
        #     tqdm.write(f"Epoch {run_epoch} val loss: {val_loss}")

        # if(run_epoch%10 == 0):
        #     model_path = "model_saves/{}/e{}_{}_{}.pth.tar".format(datetime.now().strftime("%d-%m-%y"), run_epoch, {loss_func._get_name() for loss_func in loss_function}, model.activation._get_name())
        #     torch.save({"epoch":run_epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, model_path)
        
        # with open('Epoch,Loss.csv', mode='a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([run_epoch, epoch_loss, val_loss[0], val_loss[1]])

    return epoch+1