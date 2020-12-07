import torch
from torch.utils.data import Dataset
import cv2 as cv
from pathlib import Path
from albumentations.pytorch.functional import img_to_tensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    ElasticTransform,
    RandomBrightnessContrast,
    RandomGamma,
)


def get_file_name():
    data_path = Path('root/pyb/data')
    folder = 'instrument_dataset_'
    train_image_list = []
    val_image_list = []
    for i in range(1, 9):
        current_folder = data_path / (folder + str(i + 1))
        if i < 6:
            train_image_list.extend(list((current_folder / 'images').glob('*')))
        else:
            val_image_list.extend(list((current_folder / 'images').glob('*')))

    return train_image_list, val_image_list


def load_image(path):
    img = cv.imread(path)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def load_mask(path):
    mask = cv.imread(path)
    return mask


class MyDataset(Dataset):
    def __init__(self, file_names, transform) -> None:
        super().__init__()
        self.file_names = file_names
        self.transform = transform

    def __getitem__(self, index: int):
        img_name = str(self.file_names[index])
        mask_name = img_name.replace('images', 'binary_masks')
        mask_name = mask_name.replace('jpg', 'png')
        img = load_image(img_name)
        mask = load_mask(mask_name)

        data = {'image': img, 'mask': mask}

        if self.transform is not None:
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        return img_to_tensor(img), torch.from_numpy(mask).long()

    def __len__(self) -> int:
        return len(self.file_names)
