import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

TRAIN_SIZE = 0.8


class DetectionDataset(Dataset):

    def __init__(self, data_path, config_file=None, transforms=None, split="train", data_limit=None):
        super(DetectionDataset, self).__init__()

        # первичные переменные
        self.data_path = data_path
        self.transforms = transforms
        self.split = split

        # парсинг данных
        self.image_filenames, self.mask_filenames = self._parse_root_(config_file)

        # сплит данных на трэйн и тест
        if self.split is not None:
            
            # ограничиваю объем данных, чтобы делать быстрые эксперименты
            if data_limit is None: data_limit = len(self.image_filenames) 
            assert data_limit <= len(self.image_filenames), 'data_limit could not exceed available for training data'
            
            # делаю сплит данных
            train_size = int(data_limit * TRAIN_SIZE)
            if self.split == "train":
                self.image_filenames = self.image_filenames[:train_size]
                self.mask_filenames = self.mask_filenames[:train_size]
            elif split == "val":
                self.image_filenames = self.image_filenames[train_size: data_limit]
                self.mask_filenames = self.mask_filenames[train_size: data_limit]
            else:
                raise NotImplementedError(split)

    def _parse_root_(self, config_file):
        """ парсит конфиг файл и возвращает 2 списка image_filenames, mask_filenames """
        with open(config_file, "rt") as f:
            config = json.load(f)
        image_filenames, mask_filenames = [], []
        for item in config:
            if "mask" in item:  # handling bad files during transfer
                image_filenames.append(item["file"])
                mask_filenames.append(item["mask"])

        assert len(image_filenames) == len(mask_filenames), "Images and masks lengths mismatch"
        return image_filenames, mask_filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, item):
        image_filename = os.path.join(self.data_path, self.image_filenames[item])
        mask_filename = os.path.join(self.data_path, self.mask_filenames[item])

        image = cv2.imread(image_filename).astype(np.float32) / 255. # TODO для чего делить на 255?
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image.transpose(2, 0, 1), mask # TODO для чего transpose?
