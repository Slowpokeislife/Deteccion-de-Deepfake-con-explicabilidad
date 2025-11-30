import os
import lmdb
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import timm

# ===========================
# Dataset LMDB
# ===========================
class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, subdir=False)
        with self.env.begin() as txn:
            cursor = txn.cursor()
            self.keys = [k for k, _ in cursor]

        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.env.begin() as txn:
            value = txn.get(key)

        label = int(value[0])
        img_bytes = value[1:]

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)

        if self.transform:
            pil = self.transform(pil)

        return pil, label


# ===========================
# Augmentaciones según el artículo
# ===========================
IMG_SIZE = 224

train_transform = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    T.RandomErasing(p=0.25)
])

val_transform = T.Compose([
    T.Resize(int(IMG_SIZE*1.15)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


# ===========================
# Modelo EfficientNetV2-B0
# ===========================
def create_model():
    print("Creando modelo")
    model = timm.create_model(
        "efficientnetv2_rw_m.agc_in1k",
        pretrained=True,
        num_classes=5,
        drop_path_rate=0.4  # según el paper
    )
    return model


# ===========================
# Entrenamiento
# ===========================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# ===========================
# MAIN
# ===========================
def main():
    lmdb_path = "xai_test_data_1fps.lmdb"  # <-- Ajusta si está en otro path

    dataset = LMDBImageDataset(lmdb_path, transform=train_transform)

    N = len(dataset)
    val_size = max(int(0.1 * N), 1)
    train_size = N - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-1
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0
    print("Iniciando epocas")
    for epoch in range(30):
        print(f"\n===== Epoch {epoch+1}/30 =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val  Loss: {val_loss:.4f} |  Val  Acc: {val_acc:.4f}")

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\nEntrenamiento finalizado. Mejor val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

