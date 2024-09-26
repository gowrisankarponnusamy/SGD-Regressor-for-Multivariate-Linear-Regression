# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2.Data preparation
3.Hypothesis Definition
4.cost Function 5.Parameter Update Rule 6.Iterative Training 7.Model evaluation 8.End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:GOWRISANKAR P
RegisterNumber:212222230041
*/

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())


![image](https://github.com/user-attachments/assets/820a662c-679d-4a5d-a2aa-2d1ba7ea0757)

X = df.drop(columns=['AveOccup','HousingPrice'])
X.info()

![image](https://github.com/user-attachments/assets/bdc97102-d335-455a-af6b-56f3fea2288c)

Y = df[['AveOccup','HousingPrice']]
Y.info()

![image](https://github.com/user-attachments/assets/3af24eb1-77c6-4eb4-8ebc-172e62cb6b8a)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

![image](https://github.com/user-attachments/assets/6214a796-0680-4298-8fe4-2e99b6042432)

Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

![image](https://github.com/user-attachments/assets/f150e1b2-63af-4391-9704-1a69a1082cf4)

print("\nPredictions:\n", Y_pred[:5])





```

## Output:
![image](https://github.com/user-attachments/assets/025921bc-d77b-49b5-9b61-f71dc456314e)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
