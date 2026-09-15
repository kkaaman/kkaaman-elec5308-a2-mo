import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        lane_probabilities = probabilities[:, 1]

        true_lane = (
            targets == 1
        ).float()

        intersection = (
            lane_probabilities * true_lane
        ).sum()

        denominator = (
            lane_probabilities.sum()
            + true_lane.sum()
        )

        dice = (
            2 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(
        self,
        focal_weight=1.0,
        dice_weight=1.0,
    ):
        super().__init__()

        self.focal = FocalLoss()
        self.dice = SoftDiceLoss()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        return (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
        )