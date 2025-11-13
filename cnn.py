import os
import csv
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

data_path = "/kaggle/input/deepfake-classification"
train_csv = os.path.join(data_path, "train.csv")
val_csv = os.path.join(data_path, "validation.csv")
test_csv = os.path.join(data_path, "test.csv")
train_images_path = os.path.join(data_path, "train")
val_images_path = os.path.join(data_path, "validation")
test_images_path = os.path.join(data_path, "test")

def load_labels(csv_path):
    images, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for img_id, lbl in reader:
            images.append(f"{img_id}.png")
            labels.append(int(lbl))
    return images, labels

def load_test_images(csv_path):
    images = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            images.append(f"{row[0]}.png")
    return images

train_images, train_labels = load_labels(train_csv)
val_images, val_labels = load_labels(val_csv)
test_images = load_test_images(test_csv)

class Deepfake(Dataset):
    def __init__(self, images, labels, image_path, transform=None):
        self.images = images
        self.labels = labels
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_path, self.images[idx])
        img = Image.open(path).convert("RGBA")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class TestDataset(Dataset):
    def __init__(self, images, image_path, transform=None):
        self.images = images
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_path, self.images[idx])
        img = Image.open(path).convert("RGBA")
        if self.transform:
            img = self.transform(img)
        return img, self.images[idx]

train_aug = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*4, std=[0.5]*4),
])
val_aug = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*4, std=[0.5]*4),
])
test_aug = val_aug

train_dataset = Deepfake(train_images, train_labels, train_images_path, train_aug)
val_dataset = Deepfake(val_images, val_labels, val_images_path, val_aug)
test_dataset = TestDataset(test_images, test_images_path, test_aug)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Sequential() if (in_ch == out_ch and stride == 1) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        s = self.skip(x)
        x = self.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.relu(x + s)

class Network(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.s1 = self._make_stage(64, 128, 2)
        self.s2 = self._make_stage(128, 256, 2)
        self.s3 = self._make_stage(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_stage(self, in_ch, out_ch, stride):
        return nn.Sequential(
            Block(in_ch, out_ch, stride),
            Block(out_ch, out_ch, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.pool(x)
        return self.fc(x)

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    r = np.sqrt(1 - lam)
    w = int(W * r)
    h = int(H * r)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
    x2, y2 = min(W, cx + w // 2), min(H, cy + h // 2)
    return x1, y1, x2, y2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=100,
    pct_start=0.1,
    anneal_strategy='cos'
)

best_val_acc = 0.0
for epoch in range(100):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if random.random() < 0.5:
            lam = np.random.beta(1.0, 1.0)
            idx = torch.randperm(images.size(0)).to(device)
            y1, y2 = labels, labels[idx]
            x1, y1_, x2, y2_ = rand_bbox(images.size(), lam)
            images[:, :, x1:x2, y1_:y2_] = images[idx, :, x1:x2, y1_:y2_]
            lam = 1 - ((x2 - x1) * (y2_ - y1_) / (images.size(-1) * images.size(-2)))
            out = model(images)
            loss = lam * loss_fn(out, y1) + (1 - lam) * loss_fn(out, y2)
        else:
            out = model(images)
            loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    model.eval()
    val_total, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"epoch {epoch + 1:2d}: train {train_acc:.4f}, val {val_acc:.4f}")
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

print("best acc:", best_val_acc)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

with open("submission.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image_id", "label"])
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            o1 = F.softmax(model(images), dim=1)
            o2 = F.softmax(model(torch.flip(images, [-1])), dim=1)
            out = (o1 + o2) / 2
            preds = out.argmax(1).cpu().tolist()
            for nm, p in zip(names, preds):
                wr.writerow([nm.replace(".png", ""), p])
