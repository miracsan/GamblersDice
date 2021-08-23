import torch
import torch.nn.functional as F


def probs_to_one_hot_preds(probs):
    """
    :param probs: softmax prob tensor of form (Nxd1...dnxC)
    :return: one-hot prediction tensor of form (Nxd1...dnxC)
    """
    num_classes = probs.shape[-1]
    preds = torch.argmax(probs, dim=-1)
    one_hot_preds = F.one_hot(preds, num_classes=num_classes)
    return one_hot_preds
