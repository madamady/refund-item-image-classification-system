import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri(f"sqlite:///mlflow.db")

# load data
X_train = pd.read_csv("model_data/fashion-mnist_train.csv")
X_test = pd.read_csv("model_data/fashion-mnist_test.csv")

# separate labels
y_train = X_train.pop("label").values
y_test = X_test.pop("label").values

# normalize pixels
X_train = X_train.values / 255.0
X_test = X_test.values / 255.0

# train and log
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        model, "fashion_classifier", registered_model_name="fashion_classifier"
    )

    print(f"Accuracy: {accuracy:.2f}")
