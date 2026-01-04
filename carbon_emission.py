import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('CCTS_Cement_GEI.csv')

target = "GHG_intensity_2025_26"

x = df[[
    "Baseline_Output_tonnes",
    "Baseline_GHG_intensity",
    "Baseline_Total_GHG_allowed"
    ]]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,random_state = 42)

models = {
    'LinearRegression' : LinearRegression(),
    'RandomForestRegressor' : RandomForestRegressor()
}

result = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    result[name] = {"mse": mse, "r2": r2}

for name, metrics in result.items():
    print(f"{name}:  MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}")

best_model_name = max(result, key=lambda m: result[m]["r2"])
best_model = models[best_model_name]

print("\nBest model is:", best_model_name)