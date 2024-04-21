import torch.nn as nn
import torch.nn.functional as F
import torch

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Get dimensions of input
        N, C, H, W = inputs.size()

        # Softmax the raw logits
        inputs = F.softmax(inputs, dim=1)

        # Initialize one-hot encoding and change ignore index 
        targets_mod = targets.clone()
        targets_mod[targets == self.ignore_index] = 19  # Set ignore_index to 19
        
        # Perform one-hot encoding for all classes including ignored class (19)
        targets_one_hot = F.one_hot(targets_mod, num_classes=C+1).permute(0, 3, 1, 2).float()

        # Remove the ignore class channel (19) from the one-hot encoding
        targets_one_hot = targets_one_hot[:, :C, :, :]  # Retain only the valid class channels

        # Compute the intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = (inputs * inputs).sum(dim=(2, 3)) + (targets_one_hot * targets_one_hot).sum(dim=(2, 3))
        dice_score = (2. * intersection) / (union + self.smooth)

        dice_loss = (1 - dice_score.mean())

        # Average the loss over classes and batch
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, smooth):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(ignore_index=255, smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        combined_loss = (ce_loss + dice_loss) / 2
        return combined_loss
