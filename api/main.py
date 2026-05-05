import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")

with open(os.path.join(SAVE_DIR, "classes.json")) as f:
    CLASS_NAMES = json.load(f)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model.pth"), map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Fashion classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].tolist()
    predicted = int(torch.argmax(torch.tensor(probs)))
    return {
        "label": CLASS_NAMES[predicted],
        "confidence": round(probs[predicted], 4),
        "probabilities": {CLASS_NAMES[i]: round(probs[i], 4) for i in range(len(CLASS_NAMES))}
    }
