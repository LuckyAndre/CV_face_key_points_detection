import os
from argparse import ArgumentParser

import editdistance
import numpy as np
import torch
import tqdm
from torch import optim
from torch.nn.functional import ctc_loss, log_softmax
from torch.utils.data import DataLoader

from inference_utils import get_logger
from recognition.dataset import RecognitionDataset
from recognition.model import get_model
from recognition.transforms import get_train_transforms, get_val_transforms
from recognition.utils import abc

# ПРОВЕРИЛ
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None, help="path to the data")
    parser.add_argument("--epochs", "-e", dest="epochs", type=int, help="number of train epochs", default=24)
    parser.add_argument("--batch_size", "-b", dest="batch_size", type=int, help="batch size", default=128)
    parser.add_argument("--weight_decay", "-wd", dest="weight_decay", type=float, help="weight_decay", default=1e-5)
    parser.add_argument("--lr", "-lr", dest="lr", type=float, help="lr", default=3e-4)
    parser.add_argument("--lr_step", "-lrs", dest="lr_step", type=int, help="lr step", default=None)
    parser.add_argument("--lr_gamma", "-lrg", dest="lr_gamma", type=float, help="lr gamma factor", default=None)
    parser.add_argument("--input_wh", "-wh", dest="input_wh", type=str, help="model input size", default="320x64")
    parser.add_argument("--rotation", "-r", dest="rotation", type=float, help="degree of rotation", default=10)
    parser.add_argument("--padding", "-p", dest="padding", type=float, help="padding size", default=0.1)
    parser.add_argument("--load", "-l", dest="load", type=str, help="pretrained weights", default=None)
    parser.add_argument("-o", "--output_dir", dest="output_dir", default="runs/recognition_baseline",
                        help="dir to save log and models")
    return parser.parse_args()

# ПРОВЕРИЛ
def validate(model, val_dataloader, device):
    model.eval()
    count, tp, avg_ed = 0, 0, 0

    for batch in tqdm.tqdm(val_dataloader):
        images = batch["images"].to(device)
        with torch.no_grad():
            out = model(images, decode=True) # [ГРЗ_1, ГРЗ_2,.. ГРЗ_B]
        gt = (batch["seqs"].numpy() - 1).tolist() # список B * len(ГРЗ), len(ГРЗ) может быть 8 или 9 знаков (-1 для получения индексов в алфавите кодирования)
        lens = batch["seq_lens"].numpy().tolist() # список B, список длин ГРЗ

        pos, key = 0, ''
        for i in range(len(out)): # проходимся по каждому ГРЗ
            gts = ''.join(abc[c] for c in gt[pos:pos + lens[i]]) # ground true ГРЗ
            pos += lens[i] # смещаем pos на длину ГРЗ
            if gts == out[i]: # ground true ГРЗ vs predicted ГРЗ (обе переменные в буквено-цифровом виде, как мы это видем на гос автомобильных номерах)
                tp += 1
            else:
                avg_ed += editdistance.eval(out[i], gts)
            count += 1

    acc = tp / count # доля полностью угаданных номеров
    avg_ed = avg_ed / count # среднее по батчу редакторское расстояние

    return acc, avg_ed

# ПРОВЕРИЛ
def train(model, criterion, optimizer, scheduler, train_dataloader, logger, device):
    # TODO TIP: There's always a chance to overfit to training data...
    model.train()

    epoch_losses = []
    tqdm_iter = tqdm.tqdm(train_dataloader)
    for i, batch in enumerate(tqdm_iter):
        images = batch["images"].to(device) # B x C x 64 x 320
        seqs = batch["seqs"] # вектор B * len(ГРЗ), len(ГРЗ) может быть 8 или 9 знаков
        seq_lens = batch["seq_lens"] # вектор B (содрежит длину номера ГРЗ в соответствующем элементе батча)

        """
        Модель состоит из 2х блоков: блок фичей, блок RNN.
        На входе в модель: B x C x 320 x 60.
        Блок фичей (перед RNN) создает тензор: 20 x B x 512.
        На выходе из модели: 20 х B х alphabet_size.
        """
        seqs_pred = model(images).cpu() 
        log_probs = log_softmax(seqs_pred, dim=2) # получили вероятности
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int() # вектор длины B: [20, 20, ... 20] 

        loss = criterion(log_probs, seqs, seq_lens_pred, seq_lens) # ctc_loss(input, target, input_lengths, target_lengths)
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean loss: {np.mean(epoch_losses):.4f}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) # TODO: для чего этот clip?
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    logger.info("Epoch finished! Loss: {:.5f}".format(np.mean(epoch_losses)))

    return np.mean(epoch_losses)

# ПРОВЕРИЛ
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = get_logger(os.path.join(args.output_dir, "train.log"))
    logger.info("Start training with params:")
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # TODO TIP: The provided model has _lots_ of params to tune.
    # TODO TIP: Also, it's architecture is not the only option.
    # Is recurrent part necessary at all?
    model = get_model()
    if args.load is not None:
        with open(args.load, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device)
    logger.info(f"Model type: {model.__class__.__name__}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) \
        if args.lr_step is not None else None

    # TODO TIP: Ctc_loss is not the only choice here.
    criterion = ctc_loss

    # TODO TIP: Think of data labels. What kinds of imbalance may it bring?
    # You may benefit from learning about samplers (at torch.utils.data).
    # TODO TIP: And of course given lack of data you should augment it.
    image_w, image_h = list(map(int, args.input_wh.split('x')))
    train_transforms = get_train_transforms((image_w, image_h), args.rotation, args.padding) # size, rotation, padding
    train_dataset = RecognitionDataset(args.data_path, os.path.join(args.data_path, "train_recognition.json"),
                                       abc=abc, transforms=train_transforms, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                  collate_fn=train_dataset.collate_fn) # обратить внимание: подается в collate_fn виде!

    val_transforms = get_val_transforms((image_w, image_h))
    val_dataset = RecognitionDataset(args.data_path, os.path.join(args.data_path, "train_recognition.json"),
                                     abc=abc, transforms=val_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                collate_fn=val_dataset.collate_fn) # обратить внимание: подается в collate_fn виде!

    logger.info(f"Length of train / val = {len(train_dataset)}/ {len(val_dataset)}")
    logger.info(f"Number of batches of train / val = {len(train_dataloader)} / {len(val_dataloader)}")

    best_val_acc = -1
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1} / {args.epochs}.")

        train_loss = train(model, criterion, optimizer, scheduler, train_dataloader, logger, device)
        val_acc, val_acc_ed = validate(model, val_dataloader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open(os.path.join(args.output_dir, "CP-best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)
            logger.info(f"Valid acc: {val_acc:.5f}, acc_ed: {val_acc_ed:.5f} (best)")
        else:
            logger.info(f"Valid acc: {val_acc:.5f}, acc_ed: {val_acc_ed:.5f} (best {best_val_acc:.5f})")

    with open(os.path.join(args.output_dir, "CP-last.pth"), "wb") as fp:
        torch.save(model.state_dict(), fp)


if __name__ == "__main__":
    main(parse_arguments())
