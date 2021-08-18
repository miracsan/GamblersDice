import torch
import torch.nn.functional as F

from utils import get_soft_dice_per_sample_and_class

class GenericLoss(torch.nn.Module):
    def __init__(self, alpha=1, lamda=1):
        super(GenericLoss, self).__init__()
        self.smooth = 1e-8
        self.alpha = alpha
        self.lamda = lamda
        self.ignore_bg = False


class DiceLoss(GenericLoss):
    def __init__(self, smooth=1, ignore_bg=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, outputs, targets):
        outputs = outputs.transpose(1, -1)
        targets = targets.transpose(1, -1)

        probs = torch.softmax(outputs, dim=-1)
        soft_dice_per_sample_and_class = get_soft_dice_per_sample_and_class(
            probs, targets, ignore_bg=self.ignore_bg
        )
        return 1 - torch.mean(soft_dice_per_sample_and_class)


class BaseGamblersDiceLoss(GenericLoss):
    def __init__(self, alpha, lamda):
        super(BaseGamblersDiceLoss, self).__init__(alpha, lamda)

    def forward(self, class_probs, reservations, targets):
        num_classes = targets.shape[-1]
        # because targets always have #classes
        n_dim = targets.ndim

        max_class_probs = torch.max(class_probs, dim=-1)[0]
        confident_voxels = (max_class_probs > reservations).unsqueeze(-1)

        tp = class_probs * targets

        numerator_term = 2 * (tp * confident_voxels).sum(dim=tuple(range(1, n_dim - 1)))
        denominator_term = ((class_probs + targets) * confident_voxels).sum(
            dim=tuple(range(1, n_dim - 1))
        ) + self.smooth

        dices_per_sample_and_class = numerator_term / denominator_term
        if self.ignore_bg:
            dices_per_sample_and_class = dices_per_sample_and_class[:, 1:]

        dice_loss_term = 1 - torch.mean(dices_per_sample_and_class)
        regularizer_term = torch.mean(
            torch.pow(torch.abs(reservations), self.lamda)
        )  # Take L-lamda norm
        return dice_loss_term + self.alpha * regularizer_term


class BaseCheatersDiceLoss(GenericLoss):
    def __init__(self, alpha, lamda):
        super(BaseCheatersDiceLoss, self).__init__(alpha, lamda)

    def forward(self, class_probs, reservations, targets):
        num_classes = targets.shape[-1]

        cheat_component = targets * reservations - (1 - targets) * reservations / (
            num_classes - 1
        )
        cheating_class_probs = class_probs + self.lamda * cheat_component
        cheating_dice_per_sample_and_class = get_soft_dice_per_sample_and_class(
            cheating_class_probs, targets, ignore_bg=self.ignore_bg
        )
        cheating_dice_loss_term = 1 - torch.mean(cheating_dice_per_sample_and_class)

        return cheating_dice_loss_term + self.alpha * reservations.mean()


class BaseFaultDetectorLoss(GenericLoss):
    def __init__(self):
        super(BaseFaultDetectorLoss, self).__init__()

    def forward(self, class_probs, unc_scores, targets):
        max_probs, dense_preds = torch.max(class_probs, dim=-1)
        dense_targets = torch.argmax(targets, dim=-1)

        unc_scores = torch.flatten(unc_scores)
        dense_preds = torch.flatten(dense_preds)
        dense_targets = torch.flatten(dense_targets)

        wrong_voxels = torch.ne(dense_preds, dense_targets)

        loss = torch.mean(torch.abs(wrong_voxels.float() - unc_scores))
        return loss


class BaseRankingLoss(GenericLoss):
    def __init__(self, num_sample_correct=6600, num_sample_wrong=300):
        super(BaseRankingLoss, self).__init__()
        self.num_sample_correct = num_sample_correct
        self.num_sample_wrong = num_sample_wrong

    def forward(self, class_probs, conf_scores, targets):
        device = targets.device

        max_probs, dense_preds = torch.max(class_probs, dim=-1)
        dense_targets = torch.argmax(targets, dim=-1)

        conf_scores = torch.flatten(conf_scores)
        dense_preds = torch.flatten(dense_preds)
        dense_targets = torch.flatten(dense_targets)

        correct_voxels = torch.eq(dense_preds, dense_targets)
        wrong_voxels = ~correct_voxels
        num_correct_voxels, num_wrong_voxels = correct_voxels.sum(), wrong_voxels.sum()

        loss = 0
        if num_correct_voxels > 0:
            num_sample_correct = min(self.num_sample_correct, num_correct_voxels)
            num_sample_wrong = min(self.num_sample_wrong, num_wrong_voxels)

            correct_indices = torch.where(correct_voxels)
            wrong_indices = torch.where(wrong_voxels)

            correct_scores = conf_scores[correct_indices]
            wrong_scores = conf_scores[wrong_indices]

            permuted_correct_indices = torch.randperm(num_correct_voxels)[
                :num_sample_correct
            ].to(device)
            permuted_wrong_indices = torch.randperm(num_wrong_voxels)[
                :num_sample_wrong
            ].to(device)

            sampled_correct_scores = correct_scores[permuted_correct_indices]
            sampled_wrong_scores = wrong_scores[permuted_wrong_indices]

            repeated_correct_scores = (
                sampled_correct_scores.reshape(-1, 1)
                .repeat(1, num_sample_wrong)
                .reshape(-1, 1)
            )
            repeated_wrong_scores = sampled_wrong_scores.reshape(-1, 1).repeat(
                num_sample_correct, 1
            )
            ranking_target = torch.ones(repeated_correct_scores.shape).to(device)

            loss = F.margin_ranking_loss(
                repeated_correct_scores,
                repeated_wrong_scores,
                ranking_target,
                margin=0.1,
                reduction="mean",
            )
        return loss


class RankingLoss(GenericLoss):
    """
    Adapted from https://github.com/yding5/Uncertainty-aware-training
    """

    def __init__(self, num_sample_correct=6600, num_sample_wrong=300):
        super(RankingLoss, self).__init__()
        self.base_ranking_loss = BaseRankingLoss(
            num_sample_correct=num_sample_correct, num_sample_wrong=num_sample_wrong
        )

    def forward(self, outputs, targets):
        outputs = outputs.transpose(1, -1)
        targets = targets.transpose(1, -1)

        class_probs = torch.softmax(outputs, dim=-1)
        max_probs, dense_preds = torch.max(class_probs, dim=-1)

        return self.base_ranking_loss(class_probs, max_probs, targets)


class GamblersDiceLoss(torch.nn.Module):
    def __init__(self, alpha, lamda):
        super(GamblersDiceLoss, self).__init__()
        self.base_gamblers = BaseGamblersDiceLoss(alpha, lamda)

    def forward(self, outputs, targets):
        outputs = outputs.transpose(1, -1)
        targets = targets.transpose(1, -1)

        probs = torch.softmax(outputs, dim=-1)
        class_probs, reservations = probs[..., :-1], probs[..., -1]

        return self.base_gamblers(class_probs, reservations, targets)


class CheatersDiceLoss(GenericLoss):
    def __init__(self, alpha, lamda):
        super(CheatersDiceLoss, self).__init__(alpha, lamda)
        self.base_cheaters = BaseCheatersDiceLoss(alpha, lamda)

    def forward(self, outputs, targets):
        outputs = outputs.transpose(1, -1)
        targets = targets.transpose(1, -1)

        probs = torch.softmax(outputs, dim=-1)
        class_probs, reservations = probs[..., :-1], probs[..., -1].unsqueeze(-1)

        return self.base_cheaters(class_probs, reservations, targets)


class FaultDetectorDiceLoss(GenericLoss):
    def __init__(self, alpha, lamda):
        super(FaultDetectorDiceLoss, self).__init__(alpha, lamda)
        self.dice_loss = DiceLoss()
        self.base_fault_detector_loss = BaseFaultDetectorLoss()

    def forward(self, outputs, targets):
        outputs_seg = outputs[:, :-1]
        outputs_aux = outputs[:, -1].unsqueeze(1)
        dice_loss_term = self.dice_loss(outputs_seg, targets)

        outputs_seg = outputs_seg.transpose(1, -1)
        outputs_aux = outputs_aux.transpose(1, -1)
        targets = targets.transpose(1, -1)

        class_probs = torch.softmax(outputs_seg, dim=-1)
        uncertainty_scores = torch.sigmoid(outputs_aux)

        fault_loss_term = self.base_fault_detector_loss(
            class_probs, uncertainty_scores, targets
        )
        return dice_loss_term + self.alpha * fault_loss_term


class RankingDiceLoss(GenericLoss):
    """
    Adapted from https://github.com/yding5/Uncertainty-aware-training
    """

    def __init__(self, alpha, lamda):
        super(RankingDiceLoss, self).__init__(alpha, lamda)
        self.dice_loss = DiceLoss()
        self.ranking_loss = RankingLoss()

    def forward(self, outputs, targets):
        dice_loss_term = self.dice_loss(outputs, targets)
        rank_loss_term = self.ranking_loss(outputs, targets)
        return dice_loss_term + self.alpha * rank_loss_term
