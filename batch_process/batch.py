import os
import json
import schedule
import time
import requests
import psycopg2
from datetime import datetime

with open(os.path.join(os.path.dirname(__file__), "..", "saved_model", "classes.json")) as f:
    CLASS_NAMES = json.load(f)

UPLOAD_FOLDER = "upload"
API_URL = os.getenv("API_URL", "http://api:8000/predict")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        dbname=os.getenv("DB_NAME", "refund_db"),
        user=os.getenv("DB_USER", "refund_user"),
        password=os.getenv("DB_PASSWORD", "refund_pass")
    )


def run_batch():
    print(f"[{datetime.now()}] Starting batch run...")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    files = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        print("No image files found in upload/")
        return

    conn = get_db()
    cur = conn.cursor()

    for filename in files:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "rb") as img:
            response = requests.post(API_URL, files={"file": (filename, img, "image/jpeg")})
        result = response.json()
        cur.execute(
            """INSERT INTO predictions
               (source_file, row_index, predicted_class, predicted_label, confidence, processed_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (filename, 0, CLASS_NAMES.index(result["label"]) if "label" in result else -1,
             result.get("label"), result.get("confidence"), datetime.now())
        )
        print(f"{filename}: {result.get('label')} (confidence: {result.get('confidence', 0):.2f})")

    conn.commit()
    cur.close()
    conn.close()
    print(f"[{datetime.now()}] Batch run complete.")


if os.getenv("RUN_ONCE") == "true":
    run_batch()
else:
    schedule.every().day.at("02:00").do(run_batch)
    print("Batch scheduler started. Runs nightly at 02:00.")
    while True:
        schedule.run_pending()
        time.sleep(60)
