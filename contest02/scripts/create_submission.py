# TODO TIP: Segmentation is just one of many approaches to object localization.
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import tqdm

from segmentation.models import get_model as get_segmentation_model
from inference_utils import prepare_for_segmentation, get_boxes_from_mask, prepare_for_recognition
from recognition.model import get_model as get_recognition_model

# ПРОВЕРИЛ
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None, help="path to the data")
    parser.add_argument("-t", "--seg_threshold", dest="seg_threshold", type=float, default=0.5,
                        help="decision threshold for segmentation model")
    parser.add_argument("-s", "--seg_model", dest="seg_model", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-r", "--rec_model", dest="rec_model", type=str, default=None,
                        help="path to a trained recognition model")
    parser.add_argument("--input_wh", "-wh", dest="input_wh", type=str, help="recognition model input size",
                        default="320x64")
    parser.add_argument("-o", "--output_file", dest="output_file", default="baseline_submission.csv",
                        help="file to save predictions to")
    return parser.parse_args()

# ПРОВЕРИЛ
def main(args):
    print("Start inference")
    w, h = list(map(int, args.input_wh.split('x')))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segmentation_model = get_segmentation_model()
    with open(args.seg_model, "rb") as fp:
        state_dict = torch.load(fp, map_location="cpu")
    segmentation_model.load_state_dict(state_dict)
    segmentation_model.to(device)
    segmentation_model.eval()

    recognition_model = get_recognition_model()
    with open(args.rec_model, "rb") as fp:
        state_dict = torch.load(fp, map_location="cpu")
    recognition_model.load_state_dict(state_dict)
    recognition_model.to(device)
    recognition_model.eval()

    test_images_dirname = os.path.join(args.data_path, "test")
    results = []
    files = os.listdir(test_images_dirname)
    for i, file_name in enumerate(tqdm.tqdm(files)): # предсказания идут не батчем, а отдельно для каждого номера!
        image_src = cv2.imread(os.path.join(test_images_dirname, file_name)) # H x W x 3
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB) # H x W x 3

        # 1. Segmentation.
        image, k, dw, dh = prepare_for_segmentation(image_src.astype(np.float) / 255., (512, 512)) # size_h x size_w x 3: трансформации аналогичные как при обучении
        """
        prepare_for_segmentation фактически выполняет трансформации аналогичные тем, что делались при обучении, со следующими исключениями:
        - при обучении на вход подавалось изображение, маска изображения. При инференсе - только изображение
        - при обучении на выходе изображение, маска изображения. При инференсе - изображение, k (scaling coef), dw (x pad), dh (y pad)
        Однако, на мой взгляд правильнее было бы написать единую функцию, которая принимала/возвращала разные наборы аргументов/результатов в зависимости от режима!
        """
        x = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) # 1 x 3 x size_h x size_w: изменение осей и добавление размерности батча
        with torch.no_grad():
            pred = torch.sigmoid(segmentation_model(x.to(device))).squeeze().cpu().numpy() # 1 x size_h x size_w: маска изображения, значения маски - вероятности
        mask = (pred >= args.seg_threshold).astype(np.uint8) * 255 # 1 x size_h x size_w: маска изображения, значения маски - 0 и 255

        # 2. Extraction of detected regions.
        """
        Модель сегментации выдает маску, полностью соответствующу номеру.
        По форме маска может быть параллелограммом, если номер под углом относительно нижнего края фото.
        Но процедура get_boxes_from_mask возвращает координаты ПРЯМОУГОЛЬНОЙ области минимального размера, в которую помещяется данный номер.
        """
        boxes = get_boxes_from_mask(mask, margin=0., clip=False) # np.array N x 4, где N - кол-во номеров, 4: х1, y1, х2, y2 координаты точек углов

        if len(boxes) == 0:
            results.append((file_name, []))
            continue

        # 3. Text recognition for every detected bbox.
        texts = []
        for box in boxes:
            
            # выполняю ряд обратных трансформаций, чтобы вырезать номер с исходного изображения!
            box[[0, 2]] -= dw # убираю падинг
            box[[1, 3]] -= dh
            box /= k # ресайз к исходному размеру
            box[[0, 2]] = box[[0, 2]].clip(0, image_src.shape[1] - 1) # clip на размеры изображения
            box[[1, 3]] = box[[1, 3]].clip(0, image_src.shape[0] - 1)
            box = box.astype(np.int)
            x1, y1, x2, y2 = box
            
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0: # для чего этот эксепшн, если мы сделали clip!
                raise (Exception, str(box))
                
            crop = image_src[y1: y2, x1: x2, :] # вырезаю ГРЗ из исходного изображения
            tensor = prepare_for_recognition(crop, (w, h)).to(device)
            with torch.no_grad():
                text = recognition_model(tensor, decode=True)[0]
            texts.append((x1, text))

        # all predictions must be sorted by x1
        texts.sort(key=lambda x: x[0])
        results.append((file_name, [w[1] for w in texts])) # добавляю в лист элемент вида (имя_файл, [ГРЗ1, ГРЗ2,...(обнаруженные на данном фото)])

    # Generate a submission file
    os.makedirs(f"{os.path.sep}".join(args.output_file.split(f"{os.path.sep}")[:-1]), exist_ok=True) # make folder
    with open(args.output_file, "wt") as wf:
        wf.write("file_name,plates_string\n")
        for file_name, texts in sorted(results, key=lambda x: int(os.path.splitext(x[0])[0])):
            wf.write(f"test/{file_name},{' '.join(texts)}\n")
    print('Done')


if __name__ == "__main__":
    main(parse_arguments())
