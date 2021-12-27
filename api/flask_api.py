from flask import Flask, request, jsonify
import traceback

import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "FullStack Titanic Survivor Predictor App"

@app.route('/predict', methods=['POST'])
def predict():
    if blr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            yhat = blr.predict(query)
            yhat = yhat.reshape(yhat.shape[0],)

            prediction = list(yhat)

            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('TRAIN THE MODEL FIRST')
        return( 'NO MODEL HERE TO USE')

if __name__ == '__main__':
    blr = joblib.load('model.pkl')
    print('Model loaded...')

    model_columns = joblib.load('model_columns.pkl')
    print('Model columns loaded...')

    app.run(debug=True, port=8000)