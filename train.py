import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
import tqdm

from utils import NUM_PTS #, STORE_RESULTS_PATH
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).", default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--crop-size", "-c", default=224, type=int)
    parser.add_argument("--batch-size", "-b", default=64, type=int)
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--worker", default=4, type=int)
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)
        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)

        #### calculate loss on true image size
        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        landmarks2 = landmarks.numpy().reshape((len(landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2
        landmarks2 = restore_landmarks_batch(landmarks2, fs, margins_x, margins_y).reshape((len(landmarks2), NUM_PTS * 2))  # B x NUM_PTS * 2
        pred_landmarks2 = pred_landmarks.detach().numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2
        pred_landmarks2 = restore_landmarks_batch(pred_landmarks2, fs, margins_x, margins_y).reshape((len(pred_landmarks2), NUM_PTS * 2))  # B x NUM_PTS * 2
        loss_full_size_image = loss_fn(torch.tensor(pred_landmarks2), torch.tensor(landmarks2), reduction="mean")

        loss = loss_fn(pred_landmarks, landmarks, reduction="mean") # gradient loss
        train_loss.append(loss_full_size_image.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()

        #### calculate loss on true image size
        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        landmarks2 = landmarks.numpy().reshape((len(landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2
        landmarks2 = restore_landmarks_batch(landmarks2, fs, margins_x, margins_y).reshape((len(landmarks2), NUM_PTS * 2))  # B x NUM_PTS * 2
        pred_landmarks2 = pred_landmarks.detach().numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2
        pred_landmarks2 = restore_landmarks_batch(pred_landmarks2, fs, margins_x, margins_y).reshape((len(pred_landmarks2), NUM_PTS * 2))  # B x NUM_PTS * 2
        loss_full_size_image = loss_fn(torch.tensor(pred_landmarks2), torch.tensor(landmarks2), reduction="mean")

        #loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss_full_size_image.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):

    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((args.crop_size, args.crop_size)),
        CropCenter(args.crop_size),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.0456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)), # TODO check coordinates
    ])

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.worker, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.worker, pin_memory=True,
                                shuffle=False, drop_last=False)
    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")

    print("Creating model...")
    model = models.resnext50_32x4d(pretrained=True)
    model.requires_grad_(True)
    # Меняем слой fc модели resnet18 на новый fc слой, который переобучим под нашу задачу
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.fc.requires_grad_(True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    loss_fn_train = fnn.l1_loss
    loss_fn_val = fnn.mse_loss # для валидации использую целевую метрику

    # 2. train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):

        train_loss = train(model, train_dataloader, loss_fn_train, optimizer, device=device)
        val_loss = validate(model, val_dataloader, loss_fn_val, device=device)

        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join(STORE_RESULTS_PATH, f"{args.name}_best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), train_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.worker, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join(STORE_RESULTS_PATH, f"{args.name}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join(STORE_RESULTS_PATH, f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join(STORE_RESULTS_PATH, f"{args.name}_submit.csv"))


# if __name__ == "__main__":
#     args = parse_arguments()
#     sys.exit(main(args))
