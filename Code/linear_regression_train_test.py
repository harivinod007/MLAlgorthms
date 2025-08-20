import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

dataset = pd.read_csv('SalaryData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regressor = LinearRegression()
df = pd.DataFrame({'Actual Values': X_train.ravel(),
                  'Predicted Values': y_train})
print(df)

regressor.fit(X_train, y_train)
# get the predicted values for test data
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_pred})
print(df)

mse = mean_squared_error(y_test, y_pred)
print("mse", mse)
mae = mean_absolute_error(y_test, y_pred)
print("mae", mae)
r2 = r2_score(y_test, y_pred)
print("r2", r2)
