import os
import pandas as pd
import requests
import psycopg2
from datetime import datetime

UPLOAD_FOLDER = "upload"
API_URL = os.getenv("API_URL", "http://api:8000/predict")


def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        dbname=os.getenv("DB_NAME", "refund_db"),
        user=os.getenv("DB_USER", "refund_user"),
        password=os.getenv("DB_PASSWORD", "refund_pass")
    )


def process_file(filepath, conn):
    df = pd.read_csv(filepath)
    if "label" in df.columns:
        df = df.drop("label", axis=1)

    filename = os.path.basename(filepath)
    cur = conn.cursor()

    for i, row in df.iterrows():
        pixels = row.tolist()
        response = requests.post(API_URL, json={"pixels": pixels})
        result = response.json()

        confidence = max(result["probabilities"].values())
        cur.execute(
            """INSERT INTO predictions
               (source_file, row_index, predicted_class, predicted_label, confidence, processed_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (filename, i, result["prediction"], result["label"], confidence, datetime.now())
        )
        print(f"Row {i}: {result['label']} (confidence: {confidence:.2f})")

    conn.commit()
    cur.close()


def run_batch():
    print(f"[{datetime.now()}] Starting batch run...")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".csv")]
    if not files:
        print("No CSV files found in upload/")
        return

    conn = get_db()
    for file in files:
        print(f"Processing {file}...")
        process_file(os.path.join(UPLOAD_FOLDER, file), conn)
        print(f"Done with {file}")
    conn.close()
    print(f"[{datetime.now()}] Batch run complete.")


run_batch()
