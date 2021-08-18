import numpy as np
from torch.utils.data import Dataset
import os
from utils import get_configs_from_dataset

class SegmDataset(Dataset):
    def __init__(
        self,
        path_to_processed_folder,
        fold,
        transform=None,
        target_transform=None,
        output_filenames=False,
    ):

        self.transform = transform
        self.target_transform = target_transform  # TODO: Does not currently support random transforms! Something like image, label = transform(image, label) should be used for it. Look at the VisionDataset class under torchvision : print(inspect.getsource(VisionDataset))
        self.path_to_processed_folder = path_to_processed_folder
        self.fold = fold
        self.output_filenames = output_filenames

        if fold == "train":
            train_folder = os.path.join(self.path_to_processed_folder, "train")
            self.filepaths_list = os.listdir(train_folder)
        elif fold == "val":
            val_folder = os.path.join(self.path_to_processed_folder, "val")
            self.filepaths_list = os.listdir(val_folder)
        elif fold == "test":
            test_folder = os.path.join(self.path_to_processed_folder, "test")
            self.filepaths_list = os.listdir(test_folder)

    @property
    def dataset_name(self):
        return "segmentation dataset"

    def __len__(self):
        return len(self.filepaths_list)

    def __getitem__(self, idx, output_raw=False):

        seed = np.random.randint(2147483647)

        filename = self.filepaths_list[idx]
        filepath = os.path.join(self.path_to_processed_folder, self.fold, filename)
        file = np.load(filepath, allow_pickle=True)

        image, label = file["X"], file["Y"]

        if output_raw:
            return (image, label)

        if self.transform:
            np.random.seed(seed)
            image = self.transform(image)

        if self.target_transform:
            np.random.seed(seed)
            label = self.target_transform(label)

        if self.output_filenames:
            return (image, label, filename)
        else:
            return (image, label)

    def get_class_weights(self):
        num_classes, _, _ = get_configs_from_dataset(self.dataset_name)
        total_voxels_per_class = np.zeros(num_classes, dtype=np.int)
        for idx in range(len(self)):
            _, label = self.__getitem__(idx, output_raw=True)
            total_voxels_per_class += np.array(
                [np.sum(label == i) for i in range(num_classes)]
            )

        assert (
            np.min(total_voxels_per_class) > 0
        ), "Totel # voxels should be higher than 0 for each class"
        inv_freqs = 1 / (total_voxels_per_class)
        coefs = num_classes * inv_freqs / np.sum(inv_freqs)
        return coefs


class Covid19Dataset(SegmDataset):
    @property
    def dataset_name(self):
        return "covid19"


class WHSDataset(SegmDataset):
    @property
    def dataset_name(self):
        return "whs"


class SpleenDataset(SegmDataset):
    @property
    def dataset_name(self):
        return "spleen"


class ThorDataset(SegmDataset):
    @property
    def dataset_name(self):
        return "thor"
