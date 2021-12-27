from pandas.core.arrays import categorical
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import logistic_regression

# load in the titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/timlee0131/titanic-survival-predictor/master/train.csv")

# pre-processing
include = ['Age', 'Sex', 'Embarked', 'Survived']
df = df[include]

categoricals = []
for col, col_type in df.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)

X = df_ohe[df_ohe.columns.difference(['Survived'])]
y = df_ohe['Survived']

# initialize logistic regression model and fit it
# blr = logistic_regression.BinaryLogisticRegression(eta = 0.1, iterations=100)
# blr.fit(X, y)
blr = LogisticRegression()
blr.fit(X, y)
print(blr)

# saving the model
import joblib
joblib.dump(blr, 'model.pkl')
print("model dumped!")

# loading the saved model
blr = joblib.load('model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print('Model columns dumped')