import torch
from torchvision import datasets, transforms

from customDatasets import *
from customTransforms import (
    Eye,
    NumpyToTensor,
    Standardize,
    CropToPowerOfTwo,
)


def construct_dataloaders(dataset, batch_size, num_pool_ops=3, output_filenames=False):

    if dataset == "covid19":
        """
        https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark#guidelines
        """
        # dummy_transforms = transforms.Compose([
        #        CropToPowerOfTwo(power=3),
        #        Eye(num_classes=4),
        #        NumpyToTensor(num_channels=4)
        #        ])
        # dummy_set = Covid19Dataset('data/Covid19', fold='train', transform=NumpyToTensor(), target_transform=dummy_transforms)
        # mean, std = get_dataset_mean_and_std(dummy_set, num_channels=1)
        # weights = get_class_weights(dummy_set)
        # print("mean : {0}, std:{1}".format(mean, std))
        # print(f"weights : {weights}")
        mean = [124.0994]
        std = [82.4412]

        train_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        train_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=4),
                NumpyToTensor(num_channels=4),  # There are 4 ch after the eye operation
            ]
        )

        test_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        test_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=4),
                NumpyToTensor(num_channels=4),
            ]
        )

        train_set = Covid19Dataset(
            "data/Covid19",
            fold="train",
            transform=train_transforms_image,
            target_transform=train_transforms_target,
            output_filenames=output_filenames,
        )
        val_set = Covid19Dataset(
            "data/Covid19",
            fold="val",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )
        test_set = Covid19Dataset(
            "data/Covid19",
            fold="test",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )

    elif dataset == "whs":
        """
        http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/
        """
        mean = [-795.4346]
        std = [1056.5748]

        train_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        train_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=8),
                NumpyToTensor(num_channels=8),  # There are 8 ch after the eye operation
            ]
        )

        test_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        test_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=8),
                NumpyToTensor(num_channels=8),
            ]
        )

        train_set = WHSDataset(
            "data/MM-WHS",
            fold="train",
            transform=train_transforms_image,
            target_transform=train_transforms_target,
            output_filenames=output_filenames,
        )
        val_set = WHSDataset(
            "data/MM-WHS",
            fold="val",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )
        test_set = WHSDataset(
            "data/MM-WHS",
            fold="test",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )

    elif dataset == "spleen":
        """
        From Medical Segmentation Decathlon medicaldecathlon.com/
        """
        mean = [-532.4896]
        std = [482.3780]

        train_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        train_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=2),
                NumpyToTensor(num_channels=2),  # There are 2 ch after the eye operation
            ]
        )

        test_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        test_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=2),
                NumpyToTensor(num_channels=2),
            ]
        )

        train_set = SpleenDataset(
            "data/Spleen",
            fold="train",
            transform=train_transforms_image,
            target_transform=train_transforms_target,
            output_filenames=output_filenames,
        )
        val_set = SpleenDataset(
            "data/Spleen",
            fold="val",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )
        test_set = SpleenDataset(
            "data/Spleen",
            fold="test",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )

    elif dataset == "thor":
        mean = [-731.3619]
        std = [418.0382]

        train_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        train_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=5),
                NumpyToTensor(num_channels=5),  # There are 5 ch after the eye operation
            ]
        )

        test_transforms_image = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Standardize(mean, std, num_channels=1),
                NumpyToTensor(num_channels=1),
            ]
        )
        test_transforms_target = transforms.Compose(
            [
                CropToPowerOfTwo(power=num_pool_ops),
                Eye(num_classes=5),
                NumpyToTensor(num_channels=2),
            ]
        )

        train_set = ThorDataset(
            "data/Thor",
            fold="train",
            transform=train_transforms_image,
            target_transform=train_transforms_target,
            output_filenames=output_filenames,
        )
        val_set = ThorDataset(
            "data/Thor",
            fold="val",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )
        test_set = ThorDataset(
            "data/Thor",
            fold="test",
            transform=test_transforms_image,
            target_transform=test_transforms_target,
            output_filenames=output_filenames,
        )

    else:
        print("Unknown dataset. Aborting...")
        return

    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=False, num_workers=4
    )
    dataloaders["val"] = torch.utils.data.DataLoader(
        val_set, batch_size, shuffle=False, num_workers=4
    )
    dataloaders["test"] = torch.utils.data.DataLoader(
        test_set, batch_size, shuffle=False, num_workers=4
    )

    return dataloaders


def get_dataset_mean_and_std(dataset, num_channels=3):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)
    for inputs, labels in dataloader:
        for i in range(num_channels):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
