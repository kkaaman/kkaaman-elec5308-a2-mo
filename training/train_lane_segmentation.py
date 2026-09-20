import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from autodrive.perception.lane_dataset import LaneDataset
from autodrive.perception.lane_segmentation import LaneUNet
from autodrive.perception.losses import CombinedLoss


def main():
    parser = argparse.ArgumentParser(
        description="Train the lane segmentation model."
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the prepared TuSimple dataset.",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    csv_path = data_dir + "/dataset.csv"

    image_size = (360, 640)
    batch_size = 8
    learning_rate = 3e-4
    num_epochs = 10

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Training device:", device)

    df = pd.read_csv(csv_path)

    dataset = LaneDataset(
        dataframe=df,
        data_dir=data_dir,
        image_size=image_size,
    )

    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    _test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
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
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    print("Starting training...")

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
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
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        scheduler.step()

    print("Training finished!")


if __name__ == "__main__":
    main()
