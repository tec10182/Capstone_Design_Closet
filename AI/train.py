import argparse
import json
from transformers import AutoModelForImageClassification
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


def collate_fn(batch):
    images = []
    num_images = []

    for item in batch:
        img_tensor, count = item
        images.append(img_tensor)
        num_images.append(count)

    return torch.cat(images, dim=0), num_images


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = AutoModelForImageClassification.from_pretrained(args.generator)
    embedder = Embedder(dim=args.dimension)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        embedder.load_state_dict(checkpoint["embedder_state_dict"])
    else:
        print("No checkpoint provided. Initializing a new model.")

    optimizer = optim.Adam(
        list(generator.parameters()) + list(embedder.parameters()), lr=0.001
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
        image_dir=args.image_dir, meta_dir=args.meta_dir, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    criterion = ContrastiveLoss(tau=0.5)

    generator.train()
    embedder.train()

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    wandb.init(project="Capstone_design", config=args)

    for epoch in tqdm(range(10000000)):
        epoch_loss = 0
        for idx, (images, num_crops) in enumerate(dataloader):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                images = images.to(device)
                num_crops = torch.tensor(num_crops).to(device)

                rep = generator(images).logits
                out = embedder(rep)

                print(out.shape)
                print(num_crops.shape)
                print(num_crops)

                loss = criterion(out, num_crops)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch": idx})

        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})
        early_stopping(epoch_loss)

        # 10epoch마다 모델 저장
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save, f"model_epoch_{epoch+1}.pt")
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

        if early_stopping.early_stop:
            print("Stopping training early")
            break


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-b", "--batch_size", type=int, default=42)
    args.add_argument("-da", "--domain_adaptation", type=str, default="n")
    args.add_argument("-g", "--generator", type=str, required=True)
    args.add_argument("-c", "--checkpoint", type=str, default=None)
    args.add_argument("-id", "--image_dir", type=str, required=True)
    args.add_argument("-md", "--meta_dir", type=str, required=True)
    args.add_argument("-d", "--dimension", type=int, default=64)
    args.add_argument("-s", "--save", type=str, required=True)

    args = args.parse_args()

    train(args)
