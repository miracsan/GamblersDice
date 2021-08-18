import torch.optim as optim


def construct_scheduler(scheduler_config, optimizer, last_epoch):

    if scheduler_config == "scheduler_config_1":
        lambda1 = lambda epoch: 1
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda1, last_epoch=-1
        )  # This line is necessary to initialize the scheduler
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda1, last_epoch=last_epoch
        )

    elif scheduler_config == "scheduler_config_2":
        lambda1 = lambda epoch: 0.5 ** (epoch // 75)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda1, last_epoch=-1
        )  # This line is necessary to initialize the scheduler
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda1, last_epoch=last_epoch
        )

    elif scheduler_config == "scheduler_config_3":
        lambda1 = lambda epoch: 0.5 ** (epoch // 25)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda1, last_epoch=-1
        )  # This line is necessary to initialize the scheduler
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda1, last_epoch=last_epoch
        )

    return scheduler
