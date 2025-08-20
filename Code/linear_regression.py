import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('SalaryData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(dataset.head())

plt.scatter(X, y, color="green")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (INR)")
plt.show()
