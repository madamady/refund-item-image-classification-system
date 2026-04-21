# Fashion-MNIST Refund Classifier

An online shop receives returned items daily.
Staff photograph each item, and want automatically to classify it into one of 10 clothing categories (T-shirt, Trouser, Dress, etc.) using a machine learning model trained on Fashion-MNIST.

- A **RandomForest model** is trained on clothing images and served via a **FastAPI** REST endpoint (`/predict`)
- Every night at 02:00, a **batch job** reads CSV files of pixel data from an upload folder, calls the API for each row, and stores the results (predicted label + confidence) in **PostgreSQL**
- The whole system runs on **Render** (API + cron job + database)

Architecture diagram: [docs/architecture-design.drawio](docs/architecture-design.drawio)

---

## Prerequisites

- Python 3.11+
- Docker (only for Option C — local Docker setup)
- A [Render](https://render.com) account (only for Option B — cloud deployment)
- A [Kaggle](https://www.kaggle.com) account (only if you want to train on the full dataset)

---

## How to Run

### Option A — Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python3 model/train.py

# 3. Start the API
uvicorn api.main:app --reload

# 4. Test with a sample (in a second terminal)
python3 get_sample.py | curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d @-

# 5. Run the batch manually
python3 batch_process/batch.py
```

### Option B — Render (cloud deployment)

The app is live at: https://refund-item-image-classification-system.onrender.com

To deploy your own instance:

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Blueprint**
3. Connect your GitHub repository — Render detects `render.yaml` and creates 3 services automatically:
   - **fashion-classifier-api** — Web Service (FastAPI)
   - **fashion-classifier-batch** — Cron Job (nightly at 02:00)
   - **refund-db** — PostgreSQL database
4. The build step trains the model automatically — no manual pre-training needed
5. After the API deploys, copy its public URL and set the environment variable on the batch service:
   - Go to **fashion-classifier-batch → Environment**
   - Add `API_URL = https://<your-api-url>/predict`
6. To trigger the batch job manually: **fashion-classifier-batch → Logs → Trigger Run**

### Option C — Docker (local)

> Run `python3 model/train.py` first — the API container requires `saved_model/model.pkl` at build time.

```bash
# 1. Train the model
python3 model/train.py

# 2. Start all services — API, batch, and PostgreSQL
docker-compose up --build

# API is available at http://localhost:8000
# Batch runs nightly at 02:00

# 3. Drop CSV files into the upload folder to be processed
cp model_data/fashion-mnist_train_sample.csv batch_process/upload/
```

---

## How it Works

### Training

`model/train.py` trains the classifier and saves it to `saved_model/model.pkl`.

A sample training set (`model_data/fashion-mnist_train_sample.csv`, 22,500 rows) is already included — no download needed to get started. The script uses the full dataset if present, otherwise falls back to the sample.

To train on the full 60,000-row dataset for higher accuracy:
1. Download from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)
2. Unzip and place `fashion-mnist_train.csv` and `fashion-mnist_test.csv` into `model_data/`

Steps performed by the script:
- Reads CSVs from `model_data/`, separates labels from pixel columns
- Normalizes pixel values (divides by 255)
- Trains a `RandomForestClassifier`
- Evaluates accuracy on the test set (if available)
- Saves the model to `saved_model/model.pkl`

### API

`api/main.py` loads the model once at startup and exposes:

- `GET /` — health check
- `POST /predict` — accepts a JSON body with a `pixels` array of 784 floats (28×28 image, values 0–255) and returns the predicted label and class probabilities

Example response:
```json
{
  "prediction": 9,
  "label": "Ankle boot",
  "probabilities": {
    "T-shirt/top": 0.01,
    "Ankle boot": 0.87,
    ...
  }
}
```

To test manually with a sample image:
```bash
python3 get_sample.py | curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d @-
```

### Batching

`batch_process/batch.py` processes all CSV files in the `upload/` folder:

- Sends each row to `POST /predict` with the pixel data as JSON
- Saves the result (label, confidence, timestamp) to the `predictions` table in PostgreSQL
- On Render, triggered automatically every night at 02:00 via a Cron Job

To add items for classification, drop CSV files into `batch_process/upload/`.

### Predictions Table

Results are stored in PostgreSQL:

| column | type | example |
|---|---|---|
| `id` | auto-increment integer | 1, 2, 3... |
| `source_file` | text | `returns_batch_1.csv` |
| `row_index` | integer | 0, 1, 2... |
| `predicted_class` | integer | 9 |
| `predicted_label` | text | `Ankle boot` |
| `confidence` | float | 0.87 |
| `processed_at` | timestamp | `2026-04-20 02:00:00` |

### Dataset

Each row in the CSV is a clothing image flattened to 784 pixel values (28×28 grayscale), plus a `label` column.

| label | class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The dataset does not contain real refund data — it is used only to train the image classifier.
