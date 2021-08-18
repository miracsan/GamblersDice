import torch
import torch.nn.functional as F

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def get_configs_from_dataset(dataset):
    # (num_classes, batch_size, task_type)
    config_dic = {
        "covid19": (4, 1, "segmentation"),
        "whs": (8, 1, "segmentation"),
        "spleen": (2, 1, "segmentation"),
        "thor": (5, 1, "segmentation"),
    }

    try:
        configs = config_dic[dataset]
    except:
        raise ValueError("Unknown dataset; aborting")
    return configs


def create_checkpoint(model, optimizer, method, best_acc, epoch, alias, filename):
    state = {
        "architecture": model.architecture,
        "train_method": method,
        "num_classes": model.num_classes,
        "in_channels": model.in_channels,
        "model_state_dict": model.state_dict(),
        "optim_config": optimizer.optim_config,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "epoch": epoch,
    }
    alias_path = os.path.join("results", alias)
    os.makedirs(alias_path, exist_ok=True)
    torch.save(state, os.path.join(alias_path, filename))


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


### MC


def mc_dropout_forward(model, x, p=0.5):

    with torch.no_grad():
        if model.architecture == "unet3d":
            x, skip_out64 = model.down_tr64(x)
            x = F.dropout3d(x, p=p)
            x, skip_out128 = model.down_tr128(x)
            x = F.dropout3d(x, p=p)
            x, skip_out256 = model.down_tr256(x)
            x = F.dropout3d(x, p=p)
            x, skip_out512 = model.down_tr512(x)

            x = model.up_tr256(x, skip_out256)
            x = F.dropout3d(x, p=p)
            x = model.up_tr128(x, skip_out128)
            x = F.dropout3d(x, p=p)
            x = model.up_tr64(x, skip_out64)
            x = F.dropout3d(x, p=p)

            x = model.out_tr(x)

        else:
            NotImplementedError("MC Dropout not implemented for current architecture")

    return x


def mc_dropout_confidence(model, x, p=0.5, T=100):

    repetitions = []

    for i in range(T):
        inference = F.softmax(mc_dropout_forward(model, x, p=p), dim=1)
        repetitions.append(np.float16(inference.cpu().detach().numpy()))

    repetitions = np.array(repetitions)

    mean_class_scores = np.mean(repetitions, 0)
    max_indices = np.argmax(mean_class_scores, axis=1)
    max_indices = np.expand_dims(max_indices, axis=1)
    variances = np.var(repetitions, 0)
    del repetitions
    top_class_variances = np.take_along_axis(variances, max_indices, axis=1)
    top_class_variances = np.squeeze(top_class_variances, 1)
    mean_class_scores = torch.tensor(mean_class_scores, device=x.device)

    return mean_class_scores, -top_class_variances


def get_dice_score(outputs, labels, smooth=0.0001):
    batch_size, num_classes = labels.shape[0], labels.shape[1]

    preds = torch.transpose(outputs, 1, -1).contiguous().view(-1, num_classes)
    labels = torch.transpose(labels, 1, -1).contiguous().view(-1, num_classes)

    max_indices = preds.max(dim=1)[1]

    onehot_preds = torch.zeros(
        max_indices.size(0), num_classes, device=max_indices.device
    )
    onehot_preds.scatter_(1, max_indices.unsqueeze(1), 1)
    labels = labels.type_as(onehot_preds)

    tp = (onehot_preds * labels).view(batch_size, -1, num_classes)
    onehot_preds = onehot_preds.view(batch_size, -1, num_classes)
    labels = labels.view(batch_size, -1, num_classes)

    tp_sum = torch.sum(tp, dim=1)
    onehot_preds_sum = torch.sum(onehot_preds, dim=1)
    labels_sum = torch.sum(labels, dim=1)

    numerator_term = 2 * tp_sum + smooth
    denominator_term = onehot_preds_sum + labels_sum + smooth

    mean_dice_scores = (
        torch.mean(numerator_term / denominator_term, dim=0).cpu().numpy()
    )

    return mean_dice_scores


def get_soft_dice_per_sample_and_class(probs, targets, ignore_bg=False, smooth=1):
    """
    Assumes N x D1 x D2... x C input
    """
    n_dim = targets.ndim
    tp = probs * targets
    numerator_term = 2 * tp.sum(dim=tuple(range(1, n_dim - 1)))
    denominator_term = (probs + targets).sum(dim=tuple(range(1, n_dim - 1))) + smooth

    soft_dice_per_sample_and_class = numerator_term / denominator_term
    if ignore_bg:
        soft_dice_per_sample_and_class = soft_dice_per_sample_and_class[:, 1:]

    return soft_dice_per_sample_and_class


# VISUALIZATION


def put_array_into_bins(array, num_bins):
    step_size = 1.0 / num_bins
    binned_array = np.zeros(np.shape(array), dtype=np.uint8)

    for bin_num in range(1, num_bins):
        threshold_left = np.percentile(array, 100 * bin_num * step_size)
        threshold_right = np.percentile(array, 100 * (bin_num + 1) * step_size)
        elems_in_bin = (array > threshold_left) * (array < threshold_right)
        binned_array += elems_in_bin.astype(np.uint8) * bin_num

    return binned_array


def normalize_array(array):
    max_val, min_val = np.max(array), np.min(array)
    array = (array.astype(np.float32) - min_val) / (max_val - min_val)
    return array


def extract_and_save_figures(
    save_path, filename, inputs, labels, preds, uncs, num_classes=4
):
    # Construct cmap for categorical data
    cmap = colors.ListedColormap(
        [
            "steelblue",
            "green",
            "red",
            "gold",
            "pink",
            "blue",
            "yellow",
            "brown",
            "orange",
        ]
    )
    boundaries = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    num_axial_slices = inputs.shape[0]

    for slice_num in range(0, num_axial_slices, 2):
        input_slice = inputs[slice_num]
        label_slice = labels[slice_num]
        pred_slice = preds[slice_num]
        unc_slice = uncs[slice_num]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        axes[0, 0].axis("off")
        axes[0, 0].set_title("input")
        axes[0, 0].imshow(input_slice, cmap="gray")

        axes[0, 1].axis("off")
        axes[0, 1].set_title("uncertainty")
        axes[0, 1].imshow(unc_slice)

        axes[1, 0].axis("off")
        axes[1, 0].set_title("labels")
        axes[1, 0].imshow(label_slice, cmap=cmap, norm=norm)

        axes[1, 1].axis("off")
        axes[1, 1].set_title("preds")
        axes[1, 1].imshow(pred_slice, cmap=cmap, norm=norm)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, filename + f"_slice_{slice_num}" + ".jpg"), dpi=80
        )
