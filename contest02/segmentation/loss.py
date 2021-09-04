import torch

# ПРОВЕРИЛ
def dice_coeff(input, target):
    """
    input: masks_probs.view(-1)
    target: mask_true.view(-1)
    """
    smooth = 1.
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum() # такой подсчет объяснялся в лекции Даниила про меру Жаккара
    union = input_flat.sum() + target_flat.sum()
    return (2. * intersection + smooth) / (union + smooth)

# ПРОВЕРИЛ
def dice_loss(input, target):
    # TODO TIP: Optimizing the Dice Loss usually helps segmentation a lot.
    return - torch.log(dice_coeff(input, target))
