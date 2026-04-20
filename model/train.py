import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train = pd.read_csv("model_data/fashion-mnist_train.csv")
X_test = pd.read_csv("model_data/fashion-mnist_test.csv")

y_train = X_train.pop("label").values
y_test = X_test.pop("label").values

X_train = X_train.values / 255.0
X_test = X_test.values / 255.0

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.2f}")

os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/model.pkl")
