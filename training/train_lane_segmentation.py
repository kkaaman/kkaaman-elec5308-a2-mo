import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from autodrive.perception.lane_dataset import LaneDataset
from autodrive.perception.lane_segmentation import LaneUNet
from autodrive.perception.losses import CombinedLoss


parser = argparse.ArgumentParser(
    description="Train the lane segmentation model."
)

parser.add_argument(
    "--data-dir",
    required=True,
    help="Path to the prepared TuSimple dataset.",
)

args = parser.parse_args()

DATA_DIR = args.data_dir
CSV_PATH = DATA_DIR + "/dataset.csv"

IMAGE_SIZE = (360, 640)
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Training device:", device)


df = pd.read_csv(CSV_PATH)

dataset = LaneDataset(
    dataframe=df,
    data_dir=DATA_DIR,
    image_size=IMAGE_SIZE,
)


total_size = len(dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Fixed seed keeps the dataset split reproducible.
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


model = LaneUNet(
    num_classes=2
).to(device)

criterion = CombinedLoss(
    focal_weight=1.0,
    dice_weight=1.0,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
)


print("Starting training...")

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, masks)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)

    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)
            loss = criterion(predictions, masks)

            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "lane_unet_best.pth",
        )

        print("Saved new best model!")

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f}"
    )

    scheduler.step()


print("Training finished!")