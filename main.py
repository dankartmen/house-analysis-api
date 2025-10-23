from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
import joblib  # Для сохранения моделей, если нужно

app = FastAPI()


# Загрузка данных (замените на ваш путь к CSV)
df = pd.read_csv('kc_house_data.csv', quotechar='"')  # Пример: df.columns = ['id', 'date', ...]
df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')  # Парсинг даты
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                    'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                    'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
X = df[numeric_features]  # Только числовые для базовой
y = df['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение моделей
models = {}

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
models['Linear Regression'] = lr

# 2. LASSO (alpha=1.0 по умолчанию)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
models['LASSO Regression'] = lasso

# 3. Ridge (alpha=1.0)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
models['Ridge Regression'] = ridge

# 4. Polynomial (степень 2, с LinearRegression)
poly = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
poly.fit(X_train, y_train)  # Полное обучение pipeline на raw данных
models['Polynomial Regression'] = poly

# Функция для метрик
def compute_metrics(model, X_train, y_train, X_val, y_val, model_name):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)  # Всегда используем raw данные; pipeline сам трансформирует
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    return {
        'model': model_name,
        'train': {'MSE': mse_train, 'RMSE': rmse_train, 'MAE': mae_train},
        'val': {'MSE': mse_val, 'RMSE': rmse_val, 'MAE': mae_val}
    }

# Вычисление метрик
metrics = [compute_metrics(model, X_train, y_train, X_val, y_val, name) 
           for name, model in models.items()]

# Описания моделей
descriptions = {
    'Linear Regression': 'Простая линейная модель: y = Xβ + ε. Минимизирует сумму квадратов ошибок.',
    'LASSO Regression': 'L1-регуляризация: добавляет штраф |β|, обнуляет ненужные коэффициенты (feature selection).',
    'Ridge Regression': 'L2-регуляризация: добавляет штраф β², сжимает коэффициенты, борется с мультиколлинеарностью.',
    'Polynomial Regression': 'Линейная регрессия на полиномиальных фичах (степень 2): захватывает нелинейности.'
}

@app.get("/api/metrics")
def get_metrics():
    return {
        "descriptions": descriptions,
        "metrics": metrics  # Список словарей для таблицы
    }

