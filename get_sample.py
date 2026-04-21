import os
import pandas as pd
import json

csv = (
    "model_data/fashion-mnist_test.csv"
    if os.path.exists("model_data/fashion-mnist_test.csv")
    else "model_data/fashion-mnist_train_sample.csv"
)

df = pd.read_csv(csv)
sample = df.drop("label", axis=1).iloc[0].tolist()

print(json.dumps({"pixels": sample}))
