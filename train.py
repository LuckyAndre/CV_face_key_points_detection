import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
import json
import math

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
import tqdm
#import albumentations as A
#from ipdb import set_trace

from utils import NUM_PTS
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys, TransformByKeysA
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).", default=None)
    parser.add_argument("--data-folder", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--data-size", "-d", help="Number of examples in data set", default=None)
    parser.add_argument("--crop-size", "-c", default=224, type=int)
    parser.add_argument("--batch-size", "-b", default=64, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--worker", default=4, type=int)
    return parser.parse_args()


def wing_loss(x_pred: torch.Tensor, x_true: torch.Tensor, w: float, e: float, reduction='mean'):
    
    x = x_pred - x_true # error 
    c = w - w * math.log(1 + w / e) # constant
    selector = torch.abs(x) < w # condition
    condition_true = w * torch.log(1 + torch.abs(x) / e)
    condition_false = torch.abs(x) - c
    result = torch.where(selector, condition_true, condition_false)   
    
    return torch.mean(result)


def train(model, loader, loss_fn, optimizer, device, scheduler): # loader возвращает набор всех батчей из датасета
    model.train()
    train_loss = []
    
    for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader), desc="train..."):
        # данные
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x 1942

        # прогноз и качество
        pred_landmarks = model(images).cpu()  # B x 1942 # TODO почему на CPU?
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        #loss = wing_loss(pred_landmarks, landmarks, loss_fn_w, loss_fn_e, reduction="mean")
        train_loss.append(loss.item())

        # градиентный спуск
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []

    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))

    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu() # B x 1942
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        # преобразование прогноза к размерам исходного изображения
        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    
    # folder for artefacts
    #os.makedirs(os.path.join('runs', args.name))

    # data transformator
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((args.crop_size, args.crop_size)),
        CropCenter(args.crop_size),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])
    
    # data loder
    print("Reading data ...")
    print("Temporary change train and val data (val data 20%)!")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data_folder, "train"), train_transforms, split="val", data_size=args.data_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.worker, pin_memory=True, shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data_folder, "train"), train_transforms, split="train", data_size=args.data_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.worker, pin_memory=True, shuffle=False, drop_last=False)
    
    # model
    print("Load model efficientnetv2_rw_s, which I trained on previous step ...")
    model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=2 * NUM_PTS) # efficientnet_b3
    with open(os.path.join("runs", args.name, "best_model_efficientnetv2_rw_s_l1_loss_ADAM_64000_50_lr0.001.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu") # TODO почему на CPU?
        model.load_state_dict(best_state_dict)
    model.requires_grad_(True)
    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # loss, optimizer, scheduler
    print("Tune ADAM + L1_loss ...")
    loss_fn = fnn.l1_loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate, max_lr=0.1)
    
    # train & validate
    print("Ready for training ...")
    best_val_loss = np.inf 
    metrics = {'train_time': [], 'val_time': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):

        # train
        start_time_train = datetime.now()
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device, scheduler=None)
        metrics['train_time'].append((datetime.now() - start_time_train).seconds)
        metrics['train_loss'].append(round(train_loss, 1))

        # val
        start_time_val = datetime.now()
        val_loss = validate(model, val_dataloader, fnn.mse_loss, device=device) # на валидации всегда использую L2 loss, поскольку это целевая метрика!
        metrics['val_time'].append((datetime.now() - start_time_val).seconds)
        metrics['val_loss'].append(round(val_loss, 1))

        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join('runs', args.name, f"best_model_{args.name}.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp) 
                

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data_folder, "test"), train_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.worker, pin_memory=True,
                                 shuffle=False, drop_last=False)

    # load model
    with open(os.path.join('runs', args.name, f"best_model_{args.name}.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu") # TODO почему на CPU?
        model.load_state_dict(best_state_dict)

    # save prediction
    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join('runs', args.name, f"test_predictions_{args.name}.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)
    
    # save submission
    print('Create submission...')
    create_submission(args.data_folder, test_predictions, os.path.join('runs', args.name, f"submit_{args.name}.csv"))

    # save metrics
    with open(os.path.join('runs', args.name, f"metrics_{args.name}.txt"), 'w') as outfile:
        json.dump(metrics, outfile)
        
    # save start params
    with open(os.path.join('runs', args.name, f"start_params_{args.name}.txt"), 'w') as outfile: 
        json.dump(vars(args), outfile)



# if __name__ == "__main__":
#     args = parse_arguments()
#     sys.exit(main(args))
