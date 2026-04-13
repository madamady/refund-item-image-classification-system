import os
import pandas as pd
import requests
import json
from datetime import datetime

UPLOAD_FOLDER = "upload"
RESULTS_FOLDER = "prediction-results"
API_URL = "http://127.0.0.1:8000/predict"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def process_file(filepath):
    df = pd.read_csv(filepath)

    if "label" in df.columns:
        df = df.drop("label", axis=1)

    results = []
    for i, row in df.iterrows():
        pixels = row.tolist()
        response = requests.post(API_URL, json={"pixels": pixels})
        prediction = response.json()["prediction"]
        results.append({"row": i, "prediction": prediction})
        print(f"Row {i}: prediction = {prediction}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(filepath).replace(".csv", "")
    output_path = f"{RESULTS_FOLDER}/{filename}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


for file in os.listdir(UPLOAD_FOLDER):
    if file.endswith(".csv"):
        print(f"Processing {file}...")
        process_file(os.path.join(UPLOAD_FOLDER, file))
        print(f"Done with {file}")
