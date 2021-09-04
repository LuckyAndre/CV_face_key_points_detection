"""
train.json содержит названия файла и метки (4 точки) ГРЗ.
На одном фото может быть несколько ГРЗ.
Данная процедура создает маску по размерам фото и наносит на маску области, занимаемые ГРЗ. Маска сохраняется в отдельном файле.
Также формируется конфиг файл вида: {"file": "file_name.ext", "mask": "file_name.mask.ext"}
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
    return parser.parse_args()

# ПРОВЕРИЛ
def main(args):
    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    config_segmentation = []

    for item in tqdm.tqdm(config):
        new_item = {}
        new_item["file"] = item["file"]
        image_filename = item["file"]
        image_base, ext = os.path.splitext(image_filename)
        mask_filename = image_base + ".mask" + ext # записывается в ту же папку, откуда считывается!
        nums = item["nums"] # список вида [{'box': [[794, 661], [1004, 618], [1009, 670], [799, 717]], 'text': 'M938OX116'}, {'box': [[944, 268], [995, 267], [994, 283], [942, 283]], 'text': 'H881OA116'}]

        # проверка, что файл не существует
        if os.path.exists(os.path.join(args.data_dir, mask_filename)):
            raise FileExistsError(os.path.join(args.data_dir, mask_filename))

        # считываю данные
        image = cv2.imread(os.path.join(args.data_dir, image_filename))
        if image is None:
            continue

        # создаю маску из нулей по размерам изображения
        mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)

        for num in nums: # для каждого ГРЗ на фото:
            bbox = np.asarray(num["box"])
            cv2.fillConvexPoly(mask, bbox, 255) # наносим на маску область, занимаемую ГРЗ
        cv2.imwrite(os.path.join(args.data_dir, mask_filename), mask) # сохраняю маску

        new_item["mask"] = mask_filename
        config_segmentation.append(new_item) # сохраняю данные в конфиг [{"file": "file_name.ext", "mask": "file_name.mask.ext"}, {} ...]

    output_config_filename = os.path.join(args.data_dir, "train_segmentation.json")
    with open(output_config_filename, "wt") as fp:
        json.dump(config_segmentation, fp)


if __name__ == "__main__":
    main(parse_arguments())
