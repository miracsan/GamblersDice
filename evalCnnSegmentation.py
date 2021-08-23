import os
import csv
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from utils import (
    AverageMeter,
    mc_dropout_confidence,
    normalize_array,
    extract_and_save_figures,
)


def eval_cnn(model, dataloaders, eval_method, alias, save_vars):
    (
        one_hot_predictions_list,
        labels_list,
        confidences_list,
        dimensionless_confidence_list,
    ) = get_evals_from_dataloader(
        model, dataloaders["test"], eval_method, alias, save_vars, only_conf=False
    )

    coverages = [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]

    thresholds = get_conv_thresholds_from_conf_list(
        dimensionless_confidence_list, coverages
    )
    del dimensionless_confidence_list
    _, _, _, dimensionless_val_confidence_list = get_evals_from_dataloader(
        model, dataloaders["val"], eval_method, alias, save_uncs=False, only_conf=True
    )

    logpath = os.path.join("results", alias, "log_test")
    log_exists = os.path.isfile(logpath)
    with open(logpath, "a") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        if log_exists:
            df = pd.read_csv(logpath)
            method_set = set(df["method"])

            if eval_method in method_set:
                return
        else:
            num_classes = labels_list[0].shape[2]
            logwriter.writerow(
                [
                    "method",
                    "thr_type",
                    "target_coverage",
                    "actual_coverage",
                    *[f"dice_{class_num}" for class_num in range(num_classes)],
                ]
            )

        thr_type = "test"

        for cov_num, coverage in enumerate(coverages):
            threshold = thresholds[cov_num]
            valid_coverage = np.sum(
                np.array(dimensionless_val_confidence_list) >= threshold
            ) / len(dimensionless_val_confidence_list)
            valid_coverage = np.round(valid_coverage, 3)
            dice_meter = AverageMeter()

            for i in range(len(one_hot_predictions_list)):
                one_hot_predictions = one_hot_predictions_list[i]
                labels = labels_list[i]
                confidences = confidences_list[i]
                batch_size = labels.shape[0]
                correct_predictions = one_hot_predictions * labels
                covered_idx = confidences >= threshold

                for sample_num in range(batch_size):
                    sample_covered_idx = covered_idx[sample_num]

                    tp_sum = np.sum(
                        correct_predictions[sample_num][sample_covered_idx], axis=0
                    )
                    onehot_preds_sum = np.sum(
                        one_hot_predictions[sample_num][sample_covered_idx], axis=0
                    )
                    labels_sum = np.sum(labels[sample_num][sample_covered_idx], axis=0)

                    numerator_term = 2 * tp_sum
                    denominator_term = onehot_preds_sum + labels_sum + 0.0001

                    dice_scores = numerator_term / denominator_term
                    dice_meter.update(dice_scores, 1)

            logwriter.writerow(
                [
                    eval_method,
                    thr_type,
                    np.round(valid_coverage * 100, 3),
                    np.round(coverage * 100, 3),
                    *np.round(dice_meter.avg, 4),
                ]
            )


def get_conv_thresholds_from_dataloader(model, dataloader, eval_method, coverages):
    _, _, _, dimensionless_confidence_list = get_evals_from_dataloader(
        model, dataloader, eval_method, save_uncs=False, only_conf=True
    )
    threshold_list = get_conv_thresholds_from_conf_list(
        dimensionless_confidence_list, coverages
    )
    return threshold_list


def get_conv_thresholds_from_conf_list(dimensionless_confidence_list, coverages):
    threshold_list = []
    for coverage in coverages:
        threshold = np.percentile(dimensionless_confidence_list, 100 * (1 - coverage))
        threshold_list.append(threshold)
    return threshold_list


def get_evals_from_dataloader(
    model, dataloader, eval_method, alias=None, save_uncs=False, only_conf=False
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train(False)

    (
        one_hot_predictions_list,
        labels_list,
        confidences_list,
        dimensionless_confidence_list,
    ) = ([], [], [], [])

    with torch.no_grad():
        for inputs, labels, filenames in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size, num_classes, image_shape = (
                labels.shape[0],
                labels.shape[1],
                labels.shape[2:],
            )
            n_dim = labels.dim()

            if eval_method == "softmax_response":
                class_scores = F.softmax(model(inputs)[:, :num_classes], dim=1)
                # :num_classes handles extra class architecture case
                confidences = np.max(class_scores.cpu().detach().numpy(), axis=1)

            elif eval_method == "mc_dropout":
                model_new = copy.deepcopy(model)
                model_new.train(True)
                class_scores, confidences = mc_dropout_confidence(
                    model_new, inputs, p=0.2, T=50
                )
                del model_new

            elif eval_method == "extra_class":
                probs = F.softmax(model(inputs), dim=1)
                class_scores, reservation = probs[:, :-1], probs[:, -1]
                confidences = -1 * reservation.cpu().detach().numpy()

            else:
                raise ValueError("Unknown eval method")

            desired_axis_order = (0, *range(2, n_dim), 1)
            class_scores = class_scores.permute(desired_axis_order).view(
                -1, num_classes
            )
            labels = labels.permute(desired_axis_order).view(
                batch_size, -1, num_classes
            )
            labels = labels.cpu().numpy().astype(np.uint8)

            max_indices = torch.argmax(class_scores, dim=1)
            num_voxels = class_scores.shape[0]

            one_hot_predictions = torch.zeros(
                num_voxels, num_classes, device=class_scores.device
            )
            one_hot_predictions.scatter_(1, max_indices.unsqueeze(1), 1)
            one_hot_predictions = one_hot_predictions.contiguous().view(
                batch_size, -1, num_classes
            )
            one_hot_predictions = (
                one_hot_predictions.cpu().detach().numpy().astype(np.uint8)
            )

            confidences = confidences.reshape(batch_size, -1)

            if not only_conf:
                one_hot_predictions_list.append(one_hot_predictions)
                labels_list.append(labels)
                confidences_list.append(confidences)
            dimensionless_confidence_list.extend(confidences.reshape(-1))

            if save_uncs:
                assert inputs.shape[1] == 1  # Only 1-channel input is accepted
                visualization_path = os.path.join(
                    "results", alias, "visualization", eval_method
                )

                for i in range(batch_size):
                    filename = filenames[i][:-4]
                    save_path = os.path.join(visualization_path, filename)
                    os.makedirs(save_path, exist_ok=True)

                    inputs_to_save = inputs[i][0].cpu().numpy()

                    labels_to_save = np.sum(
                        labels[i] * np.array(range(num_classes)), axis=-1
                    )
                    labels_to_save = labels_to_save.reshape(image_shape).astype(
                        np.uint8
                    )

                    one_hot_predictions_to_save = one_hot_predictions[i].astype(
                        np.uint8
                    )
                    predictions_to_save = np.sum(
                        one_hot_predictions_to_save * np.array(range(num_classes)),
                        axis=-1,
                    )
                    predictions_to_save = predictions_to_save.reshape(
                        image_shape
                    ).astype(np.uint8)

                    uncertainties_to_save = -1 * confidences[i]
                    uncertainties_to_save = uncertainties_to_save.reshape(image_shape)
                    normalized_uncertainties_to_save = normalize_array(
                        uncertainties_to_save
                    )

                    np.savez_compressed(
                        os.path.join(save_path, filename + ".npz"),
                        X=inputs_to_save,
                        labels=labels_to_save,
                        preds=predictions_to_save,
                        unc=uncertainties_to_save,
                    )

                    extract_and_save_figures(
                        save_path,
                        filename,
                        inputs_to_save,
                        labels_to_save,
                        predictions_to_save,
                        normalized_uncertainties_to_save,
                        num_classes,
                    )

    return (
        one_hot_predictions_list,
        labels_list,
        confidences_list,
        dimensionless_confidence_list,
    )
