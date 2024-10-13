import argparse
import json
from transformers import AutoModelForImageClassification, AutoConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
from model.dataset import *
import wandb
from model.augmentation import *
import torchvision.transforms as transforms
from model.embedder import *
from model.loss import *
from tqdm import tqdm
import os
from torchvision import models


def collate_fn(batch):
    images = []
    num_images = []

    for item in batch:
        img_tensor, count = item
        images.append(img_tensor)
        num_images.append(count)

    return torch.cat(images, dim=0), num_images


def validate(generator, embedder, vloader, device):
    generator.eval()
    embedder.eval()
    pos_loss = 0
    neg_loss = 0
    n = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for idx, (images, num_crops) in enumerate(vloader):
            images = images.to(device)
            num_crops = torch.tensor(num_crops).to(device)

            rep = generator(images)
            out = embedder(rep)

            loss = validate_similarity(out, num_crops)

            pos_loss += loss[0]
            neg_loss += loss[1]
            n += loss[2]

    return [pos_loss / n, neg_loss / n]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generator = AutoModelForImageClassification.from_pretrained(args.generator)
    generator = models.resnet50(pretrained=True)
    # generator = torch.nn.Sequential(*(list(generator.children())[:-1]))
    generator.fc = torch.nn.Identity()

    embedder = Embedder(dim=args.dimension)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        embedder.load_state_dict(checkpoint["embedder_state_dict"])
    else:
        print("No checkpoint provided. Initializing a new model.")

    optimizer = optim.Adam(
        list(generator.parameters()) + list(embedder.parameters()), lr=0.0001
    )

    if args.checkpoint is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    generator = generator.to(device)
    embedder = embedder.to(device)

    # generator = torch.compile(generator)
    # embedder = torch.compile(embedder)

    transform = transforms.Compose(
        [
            RandomSquareCrop(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = CustomDataset(
        image_dir=args.image_dir, meta_dir=args.train_dir, transform=transform
    )
    valset = CustomDataset(
        image_dir=args.image_dir, meta_dir=args.val_dir, transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    vloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    criterion = ContrastiveLoss(tau=0.07)

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    scaler = torch.amp.GradScaler()

    wandb.init(project="Capstone_design", config=args)

    for epoch in range(10000000):

        generator.train()
        embedder.train()

        epoch_loss = 0
        n = 0
        for idx, (images, num_crops) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}"
        ):
            # with torch.autocast(device_type=device.type, dtype=torch.float16):
            images = images.to(device)
            num_crops = torch.tensor(num_crops).to(device)

            rep = generator(images)
            out = embedder(rep)

            loss = criterion(out, num_crops)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if not torch.isfinite(loss):
                continue

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()

            epoch_loss += loss.item()
            wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch": idx})
            n += 1

        val = validate(generator, embedder, vloader, device)
        print("loss:", epoch_loss / n)

        wandb.log(
            {
                "epoch_loss": epoch_loss,
                "epoch": epoch,
                "val_pos": val[0],
                "val_neg": val[1],
            }
        )
        # early_stopping(epoch_loss / n)

        # 10epoch마다 모델 저장
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save.strip(), f"model_epoch_{epoch+1}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "embedder_state_dict": embedder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                save_path,
            )

        # if early_stopping.early_stop:
        #     print("Stopping training early")
        #     break


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-b", "--batch_size", type=int, default=42)
    args.add_argument("-da", "--domain_adaptation", type=str, default="n")
    args.add_argument("-g", "--generator", type=str, required=True)
    args.add_argument("-c", "--checkpoint", type=str, default=None)
    args.add_argument("-id", "--image_dir", type=str, required=True)
    args.add_argument("-md", "--train_dir", type=str, required=True)
    args.add_argument("-d", "--dimension", type=int, default=64)
    args.add_argument("-s", "--save", type=str, required=True)
    args.add_argument("-vd", "--val_dir", type=str, required=True)

    args = args.parse_args()

    train(args)
