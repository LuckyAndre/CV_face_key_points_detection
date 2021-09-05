"""
train.json содержит названия файла и метки (4 точки) ГРЗ.
На одном фото может быть несколько ГРЗ.
Данная процедура вырезает изображения ГРЗ из фото и сохраняет их в отдельные файлы.
Также формируется конфиг файл вида: {"file": crop_filename, "text": text}
"""

import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import tqdm

# ПРОВЕРИЛ
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", help="Path to dir containing 'train/', 'test/', 'train.json'.")
    parser.add_argument("--transform", help="If True, crop & transform box using 4 corner points;"
                                            "crop bounding box otherwise.", action="store_true")
    return parser.parse_args()


# ПРОВЕРИЛ
def get_crop(image, box):
    # TODO TIP: Maybe adding some margin could help.
    # box - это матрица 4 х 2 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] c координатами угловых точек номера
    # здесь выполняется проверка, чтобы bbox не выходила за границы изображения
    x_min = np.clip(min(box[:, 0]), 0, image.shape[1])
    x_max = np.clip(max(box[:, 0]), 0, image.shape[1])
    y_min = np.clip(min(box[:, 1]), 0, image.shape[0])
    y_max = np.clip(max(box[:, 1]), 0, image.shape[0])
    return image[y_min: y_max, x_min: x_max]


# ПРОВЕРИЛ
def main(args):
    if args.transform:
        # TODO TIP: Maybe useful to crop using corners
        # See cv2.findHomography & cv2.warpPerspective for more
        raise NotImplementedError

    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    config_recognition = []

    for item in tqdm.tqdm(config):

        image_filename = item["file"]
        image = cv2.imread(os.path.join(args.data_dir, image_filename))
        if image is None:
            continue

        image_base, ext = os.path.splitext(image_filename) # в train содержатся файлы с разным расширением: .jpg, .bmp

        nums = item["nums"]
        for i, num in enumerate(nums): # на одном фото может быть несколько номеров
            text = num["text"]
            box = np.asarray(num["box"]) # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            crop_filename = image_base + ".box" + str(i).zfill(2) + ext # str(1).zfill(2) -> '01'
            new_item = {"file": crop_filename, "text": text}
            config_recognition.append(new_item)

            if os.path.exists(crop_filename):
                continue
            else:
                crop = get_crop(image, box)
                cv2.imwrite(os.path.join(args.data_dir, crop_filename), crop)


    output_config_filename = os.path.join(args.data_dir, "train_recognition.json")
    with open(output_config_filename, "wt") as fp:
        json.dump(config_recognition, fp)


if __name__ == "__main__":
    main(parse_arguments())
