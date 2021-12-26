# FullStakc Titanic Survival Predictor

This project utilizes machine learning algorithms, namely logistic regression, to predict the survivability of a given person (user input) with certain age, sex, and cabin type information. The machine learning model is trained on the famous titanic dataset from kaggle (source: https://www.kaggle.com/hesh97/titanicdataset-traincsv) and is used as the backend engine to determine survivability. This backend engine is connected to the frontend (or postman input) via an API written with Flask.

*references*:

https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

https://github.com/eclarson/MachineLearningNotebooks/blob/master/05.%20Logistic%20Regression.ipynb

### Tech Stack
Backend Model:
* Python
* NumPy, Pandas

Backend API:
* Flask

Frontend:

## How to spin up the local server
Run `python flask_api.py` and access the API on port 8000

### Example postman body input
```JSON
[
    {"Age": 85, "Sex": "male", "Embarked": "S"}
]
```

*expected output*
```json
{
    "prediction": "[True]"
}
```