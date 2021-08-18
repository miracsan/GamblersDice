import SimpleITK as sitk
import os
import numpy as np
import torch.nn as nn
import torch
import random
import argparse
from shutil import rmtree


def partition_filenames(all_filenames, train_ratio=0.6, val_ratio=0.2, seed_num=44):
    random.seed(seed_num)
    train_filenames = random.sample(
        all_filenames, int(len(all_filenames) * train_ratio)
    )
    random.seed(seed_num)
    val_filenames = random.sample(
        list(set(all_filenames) - set(train_filenames)),
        int(len(all_filenames) * val_ratio),
    )
    test_filenames = list(
        set(all_filenames) - set(train_filenames).union(set(val_filenames))
    )
    return train_filenames, val_filenames, test_filenames


def resample(
    array,
    final_shape,
    mode="trilinear",
    input_or_label="input",
    num_classes=None,
    label_list=None,
):
    # Only HxWxD input is accepted!
    resampler = nn.Upsample(size=final_shape, mode=mode)
    if isinstance(array, np.ndarray):
        array = np.expand_dims(np.expand_dims(array, 0), 0)

    if input_or_label == "input":
        array = torch.from_numpy(array).float()
        final_array = resampler(array).numpy().reshape(final_shape)

    elif input_or_label == "label":
        array = torch.from_numpy(array)
        final_array = np.zeros(final_shape, dtype=np.uint8)

        if num_classes:
            for i in range(1, num_classes):
                binary_label = (
                    resampler((array == i).float()).numpy().reshape(final_shape) >= 0.5
                )
                final_array[binary_label] = i

        elif label_list:
            for i, label_val in enumerate(label_list):
                if i == 0:
                    continue
                binary_label = (
                    resampler((array == label_val).float()).numpy().reshape(final_shape)
                    >= 0.5
                )
                final_array[binary_label] = i

    return final_array


def prepare_covid19(dataset_dir, build_dir, build=0):
    if build:
        rmtree(build_dir, ignore_errors=True)

        train_folder = os.path.join(build_dir, "train")
        val_folder = os.path.join(build_dir, "val")
        test_folder = os.path.join(build_dir, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        desired_spacing = (6.0, 2.0, 2.0)

        scan_path = os.path.join(dataset_dir, "Scans")
        mask_path = os.path.join(dataset_dir, "Masks")

        seed_num = 44
        random.seed(seed_num)
        all_filenames = os.listdir(scan_path)
        train_filenames = random.sample(all_filenames, int(len(all_filenames) * 0.6))
        random.seed(seed_num)
        val_filenames = random.sample(
            list(set(all_filenames) - set(train_filenames)),
            int(len(all_filenames) * 0.2),
        )
        test_filenames = list(
            set(all_filenames) - set(train_filenames).union(set(val_filenames))
        )

        for mode in ["train", "val", "test"]:
            if mode == "train":
                filenames = train_filenames
                folder = train_folder
            elif mode == "val":
                filenames = val_filenames
                folder = val_folder
            elif mode == "test":
                filenames = test_filenames
                folder = test_folder

            for filename in filenames:
                scan_filepath = os.path.join(scan_path, filename)
                mask_filepath = os.path.join(mask_path, filename)
                new_filepath = os.path.join(folder, filename[:-7] + ".npz")

                scan_img = sitk.ReadImage(scan_filepath)
                scan = sitk.GetArrayFromImage(scan_img)

                if "corona" in filename:
                    scan = np.clip(scan, -1250, 250)
                    scan = (scan + 1250) / (1250 + 250) * 255

                mask_img = sitk.ReadImage(mask_filepath)
                mask = sitk.GetArrayFromImage(mask_img)

                initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
                initial_shape = np.shape(scan)

                scaling = tuple(
                    ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)
                )
                final_shape = tuple(
                    int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling)
                )

                resampled_scan = resample(scan, final_shape, input_or_label="input")
                resampled_mask = resample(
                    mask.astype(np.int16),
                    final_shape,
                    input_or_label="label",
                    num_classes=4,
                )

                np.savez_compressed(
                    new_filepath,
                    X=resampled_scan.astype(np.float32),
                    Y=resampled_mask.astype(np.uint8),
                )

    os.makedirs("data/Covid19", exist_ok=True)
    os.system(f"cp -a {build_dir} 'data/Covid19'")


def prepare_whs(dataset_dir, build_dir, build=0):
    if build:
        rmtree(build_dir, ignore_errors=True)
        # main_path = "/content/drive/My Drive/MiracGamblers/data/MM-WHS/"
        # raw_path = os.path.join(main_path, "Raw")
        # processed_path = os.path.join(main_path, "Processed")

        train_folder = os.path.join(build_dir, "train")
        val_folder = os.path.join(build_dir, "val")
        test_folder = os.path.join(build_dir, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        desired_spacing = (1.5, 1.5, 1.5)
        label_list = [0, 205, 420, 500, 550, 600, 820, 850]

        filename_list = [
            filename.strip("_label.nii.gz")
            for filename in os.listdir(dataset_dir)
            if filename.endswith("_label.nii.gz")
        ]
        train_filenames, val_filenames, test_filenames = partition_filenames(
            filename_list
        )

        for mode in ["train", "val", "test"]:
            if mode == "train":
                filenames = train_filenames
                folder = train_folder
            elif mode == "val":
                filenames = val_filenames
                folder = val_folder
            elif mode == "test":
                filenames = test_filenames
                folder = test_folder

            for filename in filenames:
                scan_filepath = os.path.join(dataset_dir, filename + "_image.nii.gz")
                mask_filepath = os.path.join(dataset_dir, filename + "_label.nii.gz")
                new_filepath = os.path.join(folder, filename + ".npz")

                scan_img = sitk.ReadImage(scan_filepath)
                scan = sitk.GetArrayFromImage(scan_img)

                mask_img = sitk.ReadImage(mask_filepath)
                mask = sitk.GetArrayFromImage(mask_img)

                initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
                initial_shape = np.shape(scan)

                scaling = tuple(
                    ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)
                )
                final_shape = tuple(
                    int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling)
                )

                resampled_scan = resample(scan, final_shape, input_or_label="input")
                resampled_mask = resample(
                    mask.astype(np.int16),
                    final_shape,
                    input_or_label="label",
                    label_list=label_list,
                )

                np.savez_compressed(
                    new_filepath,
                    X=resampled_scan.astype(np.float32),
                    Y=resampled_mask.astype(np.uint8),
                )

    os.makedirs("data/MM-WHS", exist_ok=True)
    os.system(f"cp -a {build_dir} 'data/MM-WHS'")


def prepare_spleen(dataset_dir, build_dir, build=0):
    if build:
        rmtree(build_dir, ignore_errors=True)

        train_folder = os.path.join(build_dir, "train")
        val_folder = os.path.join(build_dir, "val")
        test_folder = os.path.join(build_dir, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        desired_spacing = (5.0, 2.0, 2.0)

        scan_path = os.path.join(dataset_dir, "Scans")
        mask_path = os.path.join(dataset_dir, "Masks")

        seed_num = 44
        random.seed(seed_num)
        all_filenames = os.listdir(scan_path)
        train_filenames = random.sample(all_filenames, int(len(all_filenames) * 0.6))
        random.seed(seed_num)
        val_filenames = random.sample(
            list(set(all_filenames) - set(train_filenames)),
            int(len(all_filenames) * 0.2),
        )
        test_filenames = list(
            set(all_filenames) - set(train_filenames).union(set(val_filenames))
        )

        for mode in ["train", "val", "test"]:
            if mode == "train":
                filenames = train_filenames
                folder = train_folder
            elif mode == "val":
                filenames = val_filenames
                folder = val_folder
            elif mode == "test":
                filenames = test_filenames
                folder = test_folder

            for filename in filenames:
                scan_filepath = os.path.join(scan_path, filename)
                mask_filepath = os.path.join(mask_path, filename)
                new_filepath = os.path.join(folder, filename[:-7] + ".npz")

                scan_img = sitk.ReadImage(scan_filepath)
                scan = sitk.GetArrayFromImage(scan_img)

                mask_img = sitk.ReadImage(mask_filepath)
                mask = sitk.GetArrayFromImage(mask_img)

                initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
                initial_shape = np.shape(scan)

                scaling = tuple(
                    ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)
                )
                final_shape = tuple(
                    int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling)
                )

                resampled_scan = resample(scan, final_shape, input_or_label="input")
                resampled_mask = resample(
                    mask.astype(np.int16),
                    final_shape,
                    input_or_label="label",
                    num_classes=4,
                )

                np.savez_compressed(
                    new_filepath,
                    X=resampled_scan.astype(np.float32),
                    Y=resampled_mask.astype(np.uint8),
                )

    os.makedirs("data/Spleen", exist_ok=True)
    os.system(f"cp -a {build_dir} 'data/Spleen'")


def prepare_thor(dataset_dir, build_dir, build=0):
    if build:
        rmtree(build_dir, ignore_errors=True)
        # main_path = "/content/drive/My Drive/MiracGamblers/data/Thor/"
        # raw_path = os.path.join(main_path, "Raw")
        # processed_path = os.path.join(main_path, "Processed")

        train_folder = os.path.join(build_dir, "train")
        val_folder = os.path.join(build_dir, "val")
        test_folder = os.path.join(build_dir, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        desired_spacing = (4, 3.5, 3.5)
        label_list = [0, 1, 2, 3, 4]

        filename_list = os.listdir(dataset_dir)
        train_filenames, val_filenames, test_filenames = partition_filenames(
            filename_list
        )

        for mode in ["train", "val", "test"]:
            if mode == "train":
                filenames = train_filenames
                folder = train_folder
            elif mode == "val":
                filenames = val_filenames
                folder = val_folder
            elif mode == "test":
                filenames = test_filenames
                folder = test_folder

            for filename in filenames:
                scan_filepath = os.path.join(
                    dataset_dir, filename, filename + ".nii.gz"
                )
                mask_filepath = os.path.join(dataset_dir, filename, "GT.nii.gz")
                new_filepath = os.path.join(folder, filename + ".npz")

                scan_img = sitk.ReadImage(scan_filepath)
                scan = sitk.GetArrayFromImage(scan_img)

                mask_img = sitk.ReadImage(mask_filepath)
                mask = sitk.GetArrayFromImage(mask_img)

                initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
                initial_shape = np.shape(scan)

                scaling = tuple(
                    ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)
                )
                final_shape = tuple(
                    int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling)
                )

                resampled_scan = resample(scan, final_shape, input_or_label="input")
                resampled_mask = resample(
                    mask.astype(np.int16),
                    final_shape,
                    input_or_label="label",
                    label_list=label_list,
                )

                np.savez_compressed(
                    new_filepath,
                    X=resampled_scan.astype(np.float32),
                    Y=resampled_mask.astype(np.uint8),
                )

    os.makedirs("data/Thor", exist_ok=True)
    os.system(f"cp -a {build_dir} 'data/Thor'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", help="the dataset directory with the raw files"
    )
    parser.add_argument(
        "--build-dir", help="the directory where the processed files should be saved to"
    )
    parser.add_argument(
        "--build",
        help="process the raw files from scratch if they are not already built",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dataset", help="dataset to use: covid19, whs, spleen or thor"
    )
    options = parser.parse_args()

    if options.dataset == "covid19":
        prepare_covid19(
            dataset_dir=options.dataset_dir,
            build_dir=options.build_dir,
            build=options.build,
        )
    elif options.dataset == "whs":
        prepare_whs(
            dataset_dir=options.dataset_dir,
            build_dir=options.build_dir,
            build=options.build,
        )
    elif options.dataset == "spleen":
        prepare_spleen(
            dataset_dir=options.dataset_dir,
            build_dir=options.build_dir,
            build=options.build,
        )
    elif options.dataset == "thor":
        prepare_thor(
            dataset_dir=options.dataset_dir,
            build_dir=options.build_dir,
            build=options.build,
        )
