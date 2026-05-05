# Fashion Product Refund Classifier

An online shop receives returned items daily.
Staff photograph each item, and the system automatically classifies it into one of 10 product categories using a machine learning model trained on real fashion product images.

- A model is trained on fashion product images and served via a **FastAPI** REST endpoint (`/predict`)
- Every night at 02:00, a **batch job** picks up images from an upload folder, calls the API for each one, and stores the results (label + confidence) in **PostgreSQL**
- The whole system runs on **Render** (API + cron job + database)

Architecture diagram: [docs/architecture-design.drawio](docs/architecture-design.drawio)

---

## Prerequisites

- Python 3.11+
- Docker (only for Option C — local Docker setup)
- A [Render](https://render.com) account (only for Option B — cloud deployment)
- The [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle (only if you want to retrain the model)

---

## How to Run

### Option A — Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API 
uvicorn api.main:app --reload

# 3. Test with a sample image (in a second terminal)
curl -s -X POST http://127.0.0.1:8000/predict \
  -F "file=@model_data/archive (1)/images/10000.jpg"

# 4. Run the batch manually
python3 batch_process/batch.py
```

To retrain the model:
```bash
# 1. Download the dataset from Kaggle and unzip into model_data/
# 2. Run the training script
python3 model/train.py
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
4. The trained model is committed to the repo — no training step needed on Render
5. After the API deploys, copy its public URL and set the environment variable on the batch service:
   - Go to **fashion-classifier-batch → Environment**
   - Add `API_URL = https://<your-api-url>/predict`
6. To trigger the batch job manually: **fashion-classifier-batch → Logs → Trigger Run**

### Option C — Docker (local)

```bash
# 1. Start all services — API, batch, and PostgreSQL
docker-compose up --build

# API is available at http://localhost:8000
# Batch runs nightly at 02:00

# 2. Drop image files into the upload folder to be processed
cp model_data/"archive (1)"/images/10000.jpg batch_process/upload/
```

---

## How it Works

### Training

`model/train.py` model trained and saved to `saved_model/model.pth`.

To retrain:
1. Download [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle
2. Unzip into `model_data/` so the structure is `model_data/archive (1)/images/` and `model_data/archive (1)/styles.csv`
3. Run `python3 model/train.py`



### API

`api/main.py` loads the model once at startup and exposes:

- `GET /` — health check
- `POST /predict` — accepts an image file upload, returns the predicted category and probabilities

Example request:
```bash
curl -X POST https://refund-item-image-classification-system.onrender.com/predict \
  -F "file=@item.jpg"
```

Example response:
```json
{
  "label": "Shirts",
  "confidence": 0.85,
  "probabilities": {
    "Tshirts": 0.05,
    "Shirts": 0.85,
    "Casual Shoes": 0.01,
    ...
  }
}
```

### Batching

`batch_process/batch.py` processes all image files (`.jpg`, `.png`) in the `upload/` folder:

- Sends each image to `POST /predict`
- Saves the result (label, confidence, timestamp) to the `predictions` table in PostgreSQL
- On Render, triggered automatically every night at 02:00 via a Cron Job
- Locally (Docker), runs on a schedule inside the container

To add items for classification, drop image files into `batch_process/upload/`.

### Predictions Table

Results are stored in PostgreSQL:

| column | type | example |
|---|---|---|
| `id` | auto-increment integer | 1, 2, 3... |
| `source_file` | text | `item_photo.jpg` |
| `row_index` | integer | 0 |
| `predicted_class` | integer | 1 |
| `predicted_label` | text | `Shirts` |
| `confidence` | float | 0.85 |
| `processed_at` | timestamp | `2026-04-20 02:00:00` |

### Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) — 44,000 fashion product photos with category labels.

The model is trained on the top 10 most common article types:

| class | example items |
|---|---|
| Tshirts | casual tops |
| Shirts | formal/casual shirts |
| Casual Shoes | everyday footwear |
| Watches | wristwatches |
| Sports Shoes | athletic footwear |
| Kurtas | traditional tops |
| Tops | women's tops |
| Handbags | shoulder bags, totes |
| Heels | high-heeled footwear |
| Sunglasses | eyewear |

The dataset is not committed to the repo (too large) — only the trained model is included.
