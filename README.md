Part1: Train

1. Use the free image data set Fashion-MNIST
https://www.kaggle.com/datasets/zalando-research/fashionmnist/data
2. Download and unzip, 2CSV files: fashion-mnist_train.csv and fashion-mnist_test.csv
3. add them in model_data folder
4. create the model: create file train.py in a model/ folder
5. install dependencies from requirements.txt
6. run the train script

```
python3 model/train.py
```

7. opem the mlflow ui from browser: http://127.0.0.1:5000/


Part 2: API

- add fast API, expose endpoints
- run the API with:
```
uvicorn api.main:app --reload

```

Part 3: Create samples and call the API

- create file get_sample.py
- run the script in order to get a sample:

```
python3 get_sample.py

```
- take the output
- send the output to /predict endpoint

- create file  batch_process/batch.py



# Terminal 1 — start the API
uvicorn api.main:app --reload

# Terminal 2 — get a sample and send it to the API
python3 get_sample.py | curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @-

or
python3 get_sample.py > /tmp/sample.json

curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @/tmp/sample.json


- the get_sample.py returns a JSON object with a "pixels" array of 784 integers (28×28 pixel values, range 0–255) ( one flattened Fashion-MNIST image)



### Training

- the train.py file sets MLflow tracking URI to sqlite:///mlflow.db 
- loads the data — reads the CSVs from model_data/
- separates labels from pixels 
- normalizes and divides all pixel values 
- trains a RandomForestClassifier the 60k training images
- evaluates accuracy on the 10k test images

MLflow:
the model is registered under the name fashion_classifier version 1


### Batching

batch_process/batch.py 

- creates folders: upload/ (where new CSVs go) and prediction-results/ (where results are saved) 
- looks in upload/ for CSV files

for each file, processes it row by row:

reads the CSV into a DataFrame
drops the label column if present (as we're doing prediction, not training)
for each row, sends a POST request to http://127.0.0.1:8000/predict with the pixel data as JSON
collects the prediction from each response
saves results — writes all predictions to a timestamped JSON file in prediction-results/, e.g. fashion-mnist_test_20260413_120000.json

a) Terminal 1

uvicorn api.main:app --reload

b) add a CSV file in the upload/ folder. example:

cp model_data/fashion-mnist_test.csv batch_process/upload/

c) Terminal 2

python3 batch_process/batch.py

d) check the results 

batch_process/prediction-results/

