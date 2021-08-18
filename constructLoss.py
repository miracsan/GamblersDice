from lossFunctions import *
import torch.nn as nn


def construct_loss(method, alpha, lamda, weights, *args):
    print(f"weights : {weights}")
    print(f"{method} with alpha {alpha} and lamda {lamda}]")
    if method == "cheaters_dice":
        criterion = CheatersDiceLoss(alpha, lamda)
    elif method == "fault_dice":
        criterion = FaultDetectorDiceLoss(alpha, lamda)
    elif method == "gamblers_dice":
        criterion = GamblersDiceLoss(alpha, lamda)
    elif method == "ranking_dice":
        criterion = RankingDiceLoss(alpha, lamda)
    elif method == "dice":
        criterion = DiceLoss()
    elif method == "dice_ignore":
        criterion = DiceLoss(ignore_bg=True)
    else:
        raise NotImplementedError
        criterion = nn.CrossEntropyLoss()

    return criterion
