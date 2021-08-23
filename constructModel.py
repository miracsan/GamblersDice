import torch
from REGISTRY import extra_class_methods


def construct_model(architecture, method, num_classes, in_channels=1):
    if architecture == "unet3d":
        from model.unet3d import UNet3D

        if method in extra_class_methods:
            model = UNet3D(in_channels=in_channels, out_channels=num_classes + 1)
        else:
            model = UNet3D(in_channels=in_channels, out_channels=num_classes)

    elif architecture == "vnet":
        from model.vnet import vnet

        if method in extra_class_methods:
            model = vnet(in_channels=in_channels, out_channels=num_classes + 1)
        else:
            model = vnet(in_channels=in_channels, out_channels=num_classes)

    else:
        print("Unknown architecture. Aborting...")
        return

    model.architecture = architecture
    model.num_classes = num_classes
    model.method = method
    model.in_channels = in_channels
    return model


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    architecture = checkpoint["architecture"]
    num_classes = checkpoint["num_classes"]
    train_method = checkpoint["train_method"]
    model_state_dict = checkpoint["model_state_dict"]
    starting_epoch = checkpoint["epoch"] if "epoch" in checkpoint else 0
    best_acc = checkpoint["best_acc"] if "best_acc" in checkpoint else 0.0
    in_channels = checkpoint["in_channels"] if "in_channels" in checkpoint else 1

    model = construct_model(architecture, train_method, num_classes, in_channels)
    model.load_state_dict(model_state_dict)
    print(f"Existing model was trained using {train_method}")

    return model, starting_epoch, best_acc, train_method


def adjust_model_for_new_method(model, method):
    if model.architecture == "unet3d":
        new_model = construct_model(
            model.architecture, method, model.num_classes, model.in_channels
        )
        old_weight, old_bias = (
            model.state_dict()["out_tr.final_conv.weight"],
            model.state_dict()["out_tr.final_conv.bias"],
        )
        new_weight, new_bias = (
            new_model.state_dict()["out_tr.final_conv.weight"],
            new_model.state_dict()["out_tr.final_conv.bias"],
        )
        new_weight[:], new_bias[:] = 0, 0
        new_weight[: model.num_classes], new_bias[: model.num_classes] = (
            old_weight,
            old_bias,
        )
        model.out_tr = new_model.out_tr
        model.state_dict()["out_tr.final_conv.weight"].data.copy_(new_weight)
        model.state_dict()["out_tr.final_conv.bias"].data.copy_(new_bias)
        return model

    else:
        print("2-stage learning not implemented for given architecture. Aborting...")
        return


def put_model_to_device(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def get_num_pool_ops(architecture):
    if architecture == "vnet":
        num_pool_ops = 4
    else:
        num_pool_ops = 3
    return num_pool_ops
