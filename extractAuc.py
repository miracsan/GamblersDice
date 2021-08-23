import os
import argparse
import csv
import numpy as np
import pandas as pd


def extract_auc_scores(alias, cov_checkpoints=90, thr_type="test"):
    if type(cov_checkpoints) is int:
        cov_checkpoints = [cov_checkpoints]

    auc_logpath = os.path.join("results", alias, "auc_log_test")
    auc_log_exists = os.path.isfile(auc_logpath)

    logpath = os.path.join("results", alias, "log_test")
    df = pd.read_csv(logpath)
    df = df[df["thr_type"] == thr_type]
    num_classes = len([x for x in df.columns.tolist() if x.startswith("dice")])

    method_set = set(df["method"])

    with open(auc_logpath, "a") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        if auc_log_exists:
            extracted_df = pd.read_csv(auc_logpath)
            extracted_df = extracted_df[extracted_df["thr_type"] == thr_type]
            extracted_method_set = set(extracted_df["method"])
            method_set = method_set - extracted_method_set
        else:
            logwriter.writerow(
                [
                    "method",
                    "thr_type",
                    "end_coverage",
                    *[f"mAUC_{class_num}" for class_num in range(num_classes)],
                    "mean_foreground_mAUC",
                    "calibration_MSE",
                ]
            )

        method_list = list(method_set)
        method_list.sort()

        for method in method_list:
            method_df = (
                df[df["method"] == method].drop("method", axis=1).reset_index(drop=True)
            )
            for cov in cov_checkpoints:
                end_index = method_df[method_df["actual_coverage"] == cov].index[0]
                cov_df = method_df[: end_index + 1]

                val_coverages = cov_df["target_coverage"].to_numpy()
                coverages = cov_df["actual_coverage"].to_numpy()
                discalibration = coverages - val_coverages
                dices = cov_df[[f"dice_{x}" for x in range(num_classes)]].to_numpy()
                coverage_diffs = np.abs(coverages[1:] - coverages[:-1])
                dice_midpoints = (
                    dices[
                        1:,
                    ]
                    + dices[
                        :-1,
                    ]
                ) / 2

                areas_under_curve = coverage_diffs.reshape(-1, 1) * dice_midpoints

                sum_aucs = np.sum(areas_under_curve, axis=0)
                normalised_aucs = (
                    sum_aucs / (coverages[0] - cov) * 100
                )  # 100 is the perfect score
                normalised_aucs = np.hstack(
                    (normalised_aucs, np.mean(normalised_aucs[1:]))
                )

                logwriter.writerow(
                    [
                        method,
                        thr_type,
                        int(cov),
                        *np.round(normalised_aucs, 4),
                        np.linalg.norm(discalibration, 2),
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alias",
        help="the alias of the experiment that we want to calculate AUCs for",
        default="dice_covid19",
    )
    parser.add_argument(
        "--covs",
        help="end-coverage values for which the AUC should be calculated",
        nargs="+",
        type=int,
        default=90,
    )

    options = parser.parse_args()
    extract_auc_scores(options.alias, options.covs)
