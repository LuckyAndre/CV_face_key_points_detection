import cv2
import numpy as np

# ПРОВЕРИЛ
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask): # трансформация над изображением и маской!
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

# ПРОВЕРИЛ
class Pad(object):
    """ создает контур (паддинг) вокруг изображения (как фото-рамка) size = int(np.random.uniform(0, self.max_size) * min(w, h))"""
    def __init__(self, max_size=1.0, p=0.1):
        self.max_size = max_size
        self.p = p

    def __call__(self, image, mask):
        if np.random.uniform(0.0, 1.0) > self.p:
            return image, mask
        h, w, _ = image.shape
        size = int(np.random.uniform(0, self.max_size) * min(w, h)) # определяется размер паддинга
        image_ = cv2.copyMakeBorder(image, size, size, size, size, borderType=cv2.BORDER_CONSTANT, value=0.0)
        mask_ = cv2.copyMakeBorder(mask, size, size, size, size, borderType=cv2.BORDER_CONSTANT, value=0.0)
        return image_, mask_

# ПРОВЕРИЛ
class Crop(object):
    """ вырезаю часть изображения, размеры определяются параметрами aspect_ratio, x, y"""
    def __init__(self, min_size=0.5, min_ratio=0.5, max_ratio=2.0, p=0.25):
        self.min_size = min_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, image, mask):
        if np.random.uniform(0.0, 1.0) > self.p:
            return image, mask
        h, w, _ = image.shape
        aspect_ratio = np.random.uniform(self.min_ratio, self.max_ratio)  # = w / h
        w_ = int(np.random.uniform(self.min_size, 1.0) * w)
        h_ = int(w / aspect_ratio)
        x = np.random.randint(0, max(1, w - w_))
        y = np.random.randint(0, max(1, h - h_))
        crop_image = image[y: y + h_, x: x + w_, :]
        crop_mask = mask[y: y + h_, x: x + w_]
        return crop_image, crop_mask

# ПРОВЕРИЛ
class Resize(object):
    def __init__(self, size, keep_aspect=False):
        self.size = size
        self.keep_aspect = keep_aspect

    def __call__(self, image, mask):
        image_, mask_ = image.copy(), mask.copy()
        if self.keep_aspect:

            # определяю коэффициент сжатия
            h, w = image.shape[:2]
            k = min(self.size[0] / w, self.size[1] / h) # выбираю меньший коэффициент, т.е. по той стороне, которую сильнее потребуется сжать
            h_ = int(h * k)
            w_ = int(w * k)

            # выполняю ресайзинг
            interpolation = cv2.INTER_AREA if k <= 1 else cv2.INTER_LINEAR
            image_ = cv2.resize(image_, None, fx=k, fy=k, interpolation=interpolation)
            mask_ = cv2.resize(mask_, None, fx=k, fy=k, interpolation=interpolation)

            # по одной из сторон возникнет пустая область, которую нужно заполнить
            dh = max(0, (self.size[1] - h_) // 2)
            dw = max(0, (self.size[0] - w_) // 2)
            image_ = cv2.copyMakeBorder(image_, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=0.0)
            mask_ = cv2.copyMakeBorder(mask_, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=0.0)

        # проверка, что ресайзинг выполнен
        if image_.shape[0] != self.size[1] or image_.shape[1] != self.size[0]:
            image_ = cv2.resize(image_, self.size)
            mask_ = cv2.resize(mask_, self.size)
        return image_, mask_

# ПРОВЕРИЛ
class Flip(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, mask):
        if np.random.uniform() > self.p:
            return image, mask
        return cv2.flip(image, 1), cv2.flip(mask, 1)

# ПРОВЕРИЛ
class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        self.mean = np.asarray(mean).reshape((1, 1, 3)).astype(np.float32) # array([[[0.5, 0.5, 0.5]]])
        self.std = np.asarray(std).reshape((1, 1, 3)).astype(np.float32)

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        return image, mask


# TODO TIP: Is default image size (256) enough for segmentation of car license plates?
# TODO TIP: Chances are there's a great lib for complex data augmentations, 'Albumentations' or so...
# TODO TIP: Keywords to think about: 'class imbalance', 'lack of data'.
def get_train_transforms(image_size):
    return Compose([
        Normalize(),
        Crop(min_size=2/3, min_ratio=1.0, max_ratio=1.0, p=0.5), # если min_ratio=max_ratio, то вырезаться будет квадратное изображение, размером от 2/3 до 1 (относительно исходного)
        Flip(p=0.05), # в контексте данной задачи считаю, что flip аугментация вдоль оси Y не нужна
        Pad(max_size=0.6, p=0.25), # рамка размером от 0 до 0.6 (относительно меньшего из размеров)
        Resize(size=(image_size, image_size), keep_aspect=True)
    ])


def get_val_transforms(image_size):
    return Compose([
        Normalize(),
        Resize(size=(image_size, image_size))
    ])
