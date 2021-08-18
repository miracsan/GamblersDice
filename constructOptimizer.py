import torch
import torch.optim as optim


def construct_optimizer(
    model, optim_config="optimizer_config_1", learning_rate=0.01, weight_decay=5e-4
):
    param_iter = filter(lambda p: p.requires_grad, model.parameters())

    if optim_config == "optimizer_config_1":
        # SGD with momentum
        optimizer = optim.SGD(
            param_iter, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )

    elif optim_config == "optimizer_config_2":
        # Adam
        optimizer = optim.Adam(
            param_iter, lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay
        )

    elif optim_config == "optimizer_config_3":
        # SGD with nesterov momentum
        optimizer = optim.SGD(
            param_iter,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )

    elif optim_config == "optimizer_config_4":
        # Pure SGD - no momentum
        optimizer = optim.SGD(
            param_iter, lr=learning_rate, momentum=0, weight_decay=weight_decay
        )

    optimizer.optim_config = optim_config
    return optimizer


def load_optimizer(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    optim_config = (
        checkpoint["optimizer"] if "optimizer" in checkpoint else "optimizer_config_1"
    )

    optimizer = construct_optimizer(model, optim_config)
    optimizer.load_state_dict(optimizer_state_dict)

    return optimizer


def adjust_optimizer_for_new_hyperparams(
    optimizer, learning_rate=0.01, weight_decay=5e-4
):
    for g in optimizer.param_groups:
        g["lr"] = learning_rate
        g["weight_decay"] = weight_decay
    return optimizer
