import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_file = (
    "model_data/fashion-mnist_train.csv"
    if os.path.exists("model_data/fashion-mnist_train.csv")
    else "model_data/fashion-mnist_train_sample.csv"
)
X_train = pd.read_csv(train_file)
y_train = X_train.pop("label").values
X_train = X_train.values / 255.0

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

if os.path.exists("model_data/fashion-mnist_test.csv"):
    X_test = pd.read_csv("model_data/fashion-mnist_test.csv")
    y_test = X_test.pop("label").values
    accuracy = accuracy_score(y_test, model.predict(X_test.values / 255.0))
    print(f"Accuracy: {accuracy:.2f}")
else:
    print("Test CSV not found, skipping evaluation.")

os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/model.pkl")
