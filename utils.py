import os

import cv2 # pip install opencv-python
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils import data

np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SHARE = 0.8
DATA_SIZE = 64_000
NUM_PTS = 971
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"


# после анализа обнаружилась, что есть фото с неправильно разметкой - список таких фото в bad_files_with_bias.txt, bad_files_with_high_dispersion.txt
with open("data/bad_files_with_bias.txt") as f:
    bad_files_bias = f.read().split("\n")

with open("data/bad_files_with_high_dispersion.txt") as f:
    bad_files_dispersion = f.read().split("\n")
    

# аугментация    
class ScaleMinSideToSize(object):
    """
    Сжатие размеров изображения. Коэффициент сжатия определяется по меньшей стороне!
    """
    def __init__(self, size, elem_name='image'):
        self.size = np.asarray(size, dtype=np.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        """
        sample - это словарь:
        sample['image'] - это RGB матрица изображения
        sample['landmarks'] - матрица 971 x 2 с разметкой
        как создается sample - см. ThousandLandmarksDataset
        f - коэффициент сжатия изображения
        """
        h, w, _ = sample[self.elem_name].shape
        if h < w:
            f = self.size[1] / h
        else:
            f = self.size[0] / w

        # выполняю resize изображения
        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        # выполняю корректировку разметки по коэффициенту f
        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float() # [971, 2].reshape(-1, 2) = [971, 2]
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1) # [971, 2].reshape(-1) = [1942] (для лоса нужна размерность 1)

        return sample

    
# аугментация 
class CropCenter(object):
    def __init__(self, size, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    # определил отступы сверху и снизу
    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        # обрезка фото
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        # выполняю также корректировку разметки
        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2) # [1942].reshape(-1, 2) = [971, 2]
            # смещаю всю разметку на размеры отступов, которые я срезал
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            # [None, :] превращает tensor([margin_w, margin_h]) -> tensor([[margin_w, margin_h]])
            sample['landmarks'] = landmarks.reshape(-1)

        return sample

    
# класс для применения torchvision.transforms.<transformation_name> в пайплайне
class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample

    
# класс для применения albumentations.<transformation_name> в пайплайне
class TransformByKeysA(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(image=sample[name])['image'] # необольшое отличие в интерфейсе!
        return sample


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", data_size=DATA_SIZE, train_share=TRAIN_SHARE):
        """
        Метод записывает атрибуты
        self.landmarks - список с разметкой для каждого файла [np.array(971, 2), ... np.array(971, 2)]
        self.image_names - список с путем к файлу [path1, ... path_n]
        """
        super(ThousandLandmarksDataset, self).__init__() # вызываем __init__ метод родителя data.Dataset

        # путь к данным
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split != "test" else os.path.join(root, "test_points.csv")
        images_folder_name = os.path.join(root, "images")
        
        # атрибуты для конечного результата
        self.image_names = []
        self.landmarks = []

        # считываем данные из landmark_file_name и ограничиваем кол-во, если передан параметр data_size
        with open(landmark_file_name, "rt") as fp:
            file_data = fp.read().split("\n")[1: -1] # skip header and last empty element
        if data_size < len(file_data):
            file_data = file_data[:data_size]
            
        # обработка разметки
        for i, line in tqdm.tqdm(enumerate(file_data), total=len(file_data), desc="load landmarks..."):

            # разделение строк на train, val
            if split == "train" and i == int(train_share * len(file_data)):
                break  # reached end of train part of data
            elif split == "val" and i < int(train_share * len(file_data)):
                continue  # has not reached start of val part of data
            elements = line.strip().split("\t")
            image_name = os.path.join(images_folder_name, elements[0]) # нулевой элемент - имя файла
            
            # исключаю файлы с плохой разметкой
            if (image_name in bad_files_bias): # or (image_name in bad_files_dispersion):
                continue # skip bad files

            # сохраняю в image_names имена файлов, в landmarks разметку для соответствующего файла
            self.image_names.append(image_name)
            if split in ("train", "val"):
                landmarks = list(map(np.int, elements[1:]))
                landmarks = np.array(landmarks, dtype=np.int).reshape((len(landmarks) // 2, 2))
                # для каждого файла landmarks - это матрица рамера 971 X 2
                self.landmarks.append(landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Mетод, который читает изображение, процессит его и возвращает словарь sample со следующими ключами:
        'landmarks' # [1942 elements]
        'image' # [3 x crop x crop]
        'scale_coef' # scalar
        'crop_margin_x' # scalar
        'crop_margin_y' # scalar
        """
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks # (!) 971 х 2

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        # серия трансформаций изображения (ScaleMinSideToSize, CropCenter, ToPILImage ... определяются при запуске)
        if self.transforms is not None:
            sample = self.transforms(sample) # (!) после трансформаций sample['landmarks'] - tensor [1942]

        return sample

    def __len__(self):
        return len(self.image_names)


# def restore_landmarks(landmarks, f, margins):
#     dx, dy = margins
#     landmarks[:, 0] += dx
#     landmarks[:, 1] += dy
#     landmarks /= f
#     return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y): # landmarks shape B x NUM_PTS x 2
    landmarks[:, :, 0] += margins_x[:, None] # [:, None] преобразует вектор вида [x1, x2,...] к виду [[x1], [x2], ...]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


# TODO проверить реализацию
def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')
