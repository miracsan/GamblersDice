import numpy as np
from scipy.ndimage import affine_transform
import torch


class Eye:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, target):
        return np.eye(self.num_classes, dtype=np.uint8)[np.array(target)].astype(
            np.bool
        )


class MinMaxNormalize:
    def __init__(self, smooth=1e-4, num_channels=1):
        self.smooth = smooth
        self.num_channels = num_channels

    def __call__(self, image):

        transposed_image = np.swapaxes(image, 0, -1)
        # Assumes the last dimension of input to be channel

        image_dimensions = len(np.shape(transposed_image)) - 1
        shape_to_cast = tuple(image_dimensions * [1])

        min_vals_per_channel = np.array(list(map(np.min, transposed_image))).reshape(
            -1, *shape_to_cast
        )
        # Mins of each channel (axis 0)
        max_vals_per_channel = np.array(list(map(np.max, transposed_image))).reshape(
            -1, *shape_to_cast
        )

        normalized_transposed_image = (transposed_image - min_vals_per_channel) / (
            max_vals_per_channel - min_vals_per_channel + self.smooth
        )
        normalized_image = np.swapaxes(normalized_transposed_image, 0, -1)

        return normalized_image


class Standardize:
    def __init__(self, mean, std, smooth=1e-4, num_channels=1):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.smooth = smooth
        self.num_channels = num_channels

    def __call__(self, image):
        if self.num_channels > 1:
            transposed_image = np.swapaxes(image, 0, -1)
            # Assumes the last dimension of input to be channel

            image_dimensions = len(np.shape(transposed_image)) - 1
            shape_to_cast = tuple(image_dimensions * [1])

            mean_vals_per_channel = self.mean.reshape(-1, *shape_to_cast)
            std_vals_per_channel = self.std.reshape(-1, *shape_to_cast)

            standardized_transposed_image = (
                transposed_image - mean_vals_per_channel
            ) / (std_vals_per_channel + self.smooth)
            standardized_image = np.swapaxes(standardized_transposed_image, 0, -1)

        else:
            standardized_image = (image - self.mean) / (self.std + self.smooth)

        return standardized_image


class CenterCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        if len(self.crop_size) == 3:
            height, width, depth = image.shape[:3]
            height_crop, width_crop, depth_crop = (
                self.crop_size[0],
                self.crop_size[1],
                self.crop_size[2],
            )

            if height_crop != "all":
                height_min, height_max = max(height // 2 - height_crop // 2, 0), min(
                    height // 2 + height_crop // 2, height
                )
            else:
                height_min, height_max = (0, height)

            if width_crop != "all":
                width_min, width_max = max(width // 2 - width_crop // 2, 0), min(
                    width // 2 + width_crop // 2, width
                )
            else:
                width_min, width_max = (0, width)

            if depth_crop != "all":
                depth_min, depth_max = max(depth // 2 - depth_crop // 2, 0), min(
                    depth // 2 + depth_crop // 2, depth
                )
            else:
                depth_min, depth_max = (0, depth)

            cropped_image = image[
                height_min:height_max, width_min:width_max, depth_min:depth_max, ...
            ]
        elif len(self.crop_size) == 2:
            height, width = image.shape[:2]
            height_crop, width_crop = self.crop_size[0], self.crop_size[1]

            if height_crop != "all":
                height_min, height_max = max(height // 2 - height_crop // 2, 0), min(
                    height // 2 + height_crop // 2, height
                )
            else:
                height_min, height_max = (0, height)

            if width_crop != "all":
                width_min, width_max = max(width // 2 - width_crop // 2, 0), min(
                    width // 2 + width_crop // 2, width
                )
            else:
                width_min, width_max = (0, width)

            cropped_image = image[height_min:height_max, width_min:width_max, ...]
        else:
            raise ValueError("Unknown crop dimensions")

        return cropped_image


class CropToPowerOfTwo:
    def __init__(self, power=3):
        self.divisor = 2 ** power

    def __call__(self, image):
        initial_shape = image.shape[:3]
        desired_shape = tuple(elem - elem % self.divisor for elem in initial_shape)

        cropper = CenterCrop(desired_shape)
        cropped_image = cropper(image)
        return cropped_image


class NumpyToTensor:
    def __init__(self, smooth=1e-4, num_channels=1):
        self.smooth = smooth
        self.num_channels = num_channels

    def __call__(self, array):
        if self.num_channels > 1:
            new_axes = np.roll(np.array(range(array.ndim)), 1)
            array = np.transpose(array, new_axes)

        else:
            array = np.expand_dims(array, 0)

        tensor = torch.Tensor(array)
        return tensor


class GetOneChannel:
    def __init__(self, channel_index=0):
        self.channel_index = channel_index

    def __call__(self, array):
        new_array = array[..., self.channel_index]
        return new_array


class RandomRotate:
    # SHOULD COME BEFORE THE EYE OPERATION FOR LABELS
    def __call__(self, array):
        ndim = array.ndim
        m = np.eye(ndim, dtype=np.float32) + np.random.uniform(-0.2, 0.2, (ndim, ndim))
        array = affine_transform(array, m, order=0)
        return array


class RandomAxisFlip:
    # SHOULD COME BEFORE THE EYE OPERATION FOR LABELS
    def __call__(self, array):
        new_dims = np.random.permutation(tuple(range(array.ndim)))
        array = np.transpose(array, new_dims)
        return array
