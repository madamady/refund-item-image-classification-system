import os
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

app = FastAPI()

# load the model from the local mlruns artifact path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")
model = mlflow.sklearn.load_model("models:/fashion_classifier/1")


class ImageData(BaseModel):
    pixels: List[float]


@app.get("/")
def root():
    return {"message": "Fashion classifier API is running"}


@app.get("/hello")
def root():
    return {"message": "Hello from the fashion classifier API!"}


@app.post("/predict")
def predict(data: ImageData):
    pixels = pd.DataFrame([data.pixels])
    normalized = pixels / 255.0
    prediction = model.predict(normalized)
    probabilities = model.predict_proba(normalized)[0].tolist()
    predicted_class = int(prediction[0])
    return {
        "prediction": predicted_class,
        "label": CLASS_NAMES[predicted_class],
        "probabilities": {CLASS_NAMES[i]: round(probabilities[i], 4) for i in range(len(CLASS_NAMES))}
    }
