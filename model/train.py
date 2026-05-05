import os
import json
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score

DATA_DIR = "model_data/archive (1)"
IMG_DIR = os.path.join(DATA_DIR, "images")
STYLES_CSV = os.path.join(DATA_DIR, "styles.csv")
SAVE_DIR = "saved_model"
SAMPLES_PER_CLASS = 200
EPOCHS = 3
BATCH_SIZE = 32

TOP10 = [
    "Tshirts", "Shirts", "Casual Shoes", "Watches", "Sports Shoes",
    "Kurtas", "Tops", "Handbags", "Heels", "Sunglasses"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FashionDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_id, label = self.records[idx]
        img = Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg")).convert("RGB")
        return transform(img), label


def build_records(df):
    records = []
    for label_idx, cls in enumerate(TOP10):
        ids = df[df["articleType"] == cls]["id"].tolist()[:SAMPLES_PER_CLASS]
        for id_ in ids:
            if os.path.exists(os.path.join(IMG_DIR, f"{id_}.jpg")):
                records.append((id_, label_idx))
    return records


df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
records = build_records(df)
random.shuffle(records)

split = int(0.8 * len(records))
train_loader = DataLoader(FashionDataset(records[:split]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FashionDataset(records[split:]), batch_size=BATCH_SIZE)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, len(TOP10))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    acc = accuracy_score(targets, preds)
    print(f"Epoch {epoch+1}/{EPOCHS} — val accuracy: {acc:.2f}")

os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))
with open(os.path.join(SAVE_DIR, "classes.json"), "w") as f:
    json.dump(TOP10, f)
print("Model saved.")
