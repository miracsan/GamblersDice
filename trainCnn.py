import sys
import csv
import time

from shutil import rmtree
from torch.autograd import Variable
from utils import *
from dataloaders import *
from constructModel import *
from constructLoss import construct_loss
from constructOptimizer import (
    construct_optimizer,
    load_optimizer,
    adjust_optimizer_for_new_hyperparams,
)
from constructScheduler import construct_scheduler

from torch.cuda.amp import autocast, GradScaler


def train_cnn(
    method,
    architecture,
    alias,
    dataset,
    pretrain,
    num_epochs,
    learning_rate,
    weight_decay,
    alpha,
    lamda,
    impatience,
    optimizer_config,
    scheduler_config,
    find_new_best,
    use_model,
):

    num_classes, batch_size, task_type = get_configs_from_dataset(dataset)
    num_pool_ops = get_num_pool_ops(architecture)

    if use_model:
        model, starting_epoch, best_acc, prev_method = load_model(use_model)
        if method != prev_method:
            model = adjust_model_for_new_method(model, method, architecture)
        model = put_model_to_device(
            model
        )  # Model needs to be put into device before constructing the optimizer

        if method == prev_method:
            optimizer = load_optimizer(model, use_model)
            optimizer = adjust_optimizer_for_new_hyperparams(optimizer)
        else:
            optimizer = construct_optimizer(
                model, optimizer_config, learning_rate, weight_decay
            )

        with open(os.path.join("results", alias, "log_train"), "a") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["LEARNING_RATE : ", learning_rate])

    else:
        model = construct_model(architecture, method, num_classes)
        model = put_model_to_device(model)
        optimizer = construct_optimizer(
            model, optimizer_config, learning_rate, weight_decay
        )
        starting_epoch = 0
        best_acc = 0.0

        rmtree(os.path.join("results", alias), ignore_errors=True)
        os.makedirs(os.path.join("results", alias))

        with open(os.path.join("results", alias, "log_train"), "a") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["METHOD : ", method])
            logwriter.writerow(["DATASET : ", dataset])
            logwriter.writerow(["ARCHITECTURE : ", architecture])
            logwriter.writerow(["NUM_EPOCHS : ", num_epochs])
            logwriter.writerow(["LEARNING_RATE : ", learning_rate])
            logwriter.writerow(["WEIGHT_DECAY : ", weight_decay])
            logwriter.writerow(["PRETRAIN : ", pretrain])
            logwriter.writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
            )

    # Construct datalaoders, loss, optimizer and scheduler
    dataloaders = construct_dataloaders(
        dataset, batch_size, num_pool_ops, output_filenames=True
    )
    criterion = construct_loss(
        method, alpha, lamda, dataloaders["train"].dataset.get_class_weights()
    )
    scheduler = construct_scheduler(scheduler_config, optimizer, starting_epoch)

    # train model
    since = time.time()

    if pretrain:
        model = pretrain_model(
            architecture, model, method, dataloaders, optimizer, scheduler, alias=alias
        )

    model = train_model(
        model,
        method,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        starting_epoch,
        num_epochs,
        impatience,
        alias=alias,
        best_acc=best_acc,
        find_new_best=find_new_best,
    )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )


def train_model(
    model,
    method,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    starting_epoch,
    num_epochs,
    impatience_limit,
    alias,
    best_acc=0.0,
    find_new_best=False,
):
    impatience = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scaler = GradScaler()

    dataset_sizes = {
        x: len(dataloaders[x]) * dataloaders[x].batch_size for x in ["train", "val"]
    }

    print(f"Training until the end of epoch {starting_epoch + num_epochs}")

    # iterate over epochs
    for epoch in range(starting_epoch + 1, starting_epoch + num_epochs + 1):
        print(f"Epoch {epoch}/{starting_epoch + num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            accMeter = AverageMeter()
            lossMeter = AverageMeter()

            # iterate over all data in train/val dataloader:
            for batch_num, (inputs, labels, _) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size, num_classes = labels.shape[0], labels.shape[1]

                inputs, labels = map(Variable, (inputs, labels))

                with torch.set_grad_enabled(phase == "train"):
                    optimizer.zero_grad()

                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if loss != loss:
                            raise ValueError("NaN encountered during training")

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                accuracy = (
                    get_dice_score(outputs[:, :num_classes], labels) * 100
                )  # num_classes is used to handle extra class case

                lossMeter.update(loss.item(), batch_size)
                accMeter.update(accuracy, batch_size)

                sys.stdout.write(
                    "\r Progress in the epoch:     %.3f"
                    % ((batch_num + 1) * batch_size / dataset_sizes[phase] * 100)
                )  # keep track of the progress
                sys.stdout.flush()

            epoch_loss = lossMeter.avg
            epoch_acc = accMeter.avg

            if phase == "train":
                last_train_loss = epoch_loss
                last_train_acc = epoch_acc

            # checkpoint model if has best val loss yet
            elif phase == "val":
                # Create checkpoint in each epoch
                create_checkpoint(
                    model, optimizer, method, best_acc, epoch, alias, "checkpoint"
                )

                # Create checkpoint for the best model
                if np.mean(epoch_acc) > np.mean(best_acc) or find_new_best:
                    find_new_best = 0
                    impatience = 0
                    best_acc, best_loss, best_epoch = epoch_acc, epoch_loss, epoch
                    create_checkpoint(
                        model,
                        optimizer,
                        method,
                        best_acc,
                        epoch,
                        alias,
                        "best_checkpoint",
                    )

                else:
                    impatience += 1

                # log training and validation loss over each epoch
                with open(os.path.join("results", alias, "log_train"), "a") as logfile:
                    logwriter = csv.writer(logfile, delimiter=",")
                    logwriter.writerow(
                        [epoch, last_train_loss, last_train_acc, epoch_loss, epoch_acc]
                    )

            print(
                phase
                + " epoch {}:loss {:.4f} acc {} with data size {}".format(
                    epoch, epoch_loss, epoch_acc, dataset_sizes[phase]
                )
            )

            if phase == "val":
                print(f"impatience : {impatience}")

            if impatience == impatience_limit:
                return model

        scheduler.step()

    return model
