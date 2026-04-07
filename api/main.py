import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# load the model from MLflow
model = mlflow.sklearn.load_model("models:/fashion_classifier/1")  # MLflow model URI
# models:/<model-name>/<version>


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
    print("Hello from predict endpoint!")

    # pixels = pd.DataFrame([data.pixels])
    # prediction = model.predict(pixels / 255.0)
    # return {"prediction": int(prediction[0])}
