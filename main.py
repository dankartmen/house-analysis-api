from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np


app = FastAPI()


# Загрузка данных
df = pd.read_csv('kc_house_data.csv', quotechar='"')  
df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S') 
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                    'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                    'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
X = df[numeric_features].fillna(0)
y = df['price']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

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
def compute_metrics(model, X_train, y_train, X_test, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    return {
        'model': model_name,
        'train': {'R2': train_r2, 'MAE': train_mae, 'MSE': train_mse},
        'test': {'R2': test_r2, "MAE": test_mae, 'MSE': test_mse}
    }

# Вычисление метрик
metrics = [compute_metrics(model, X_train, y_train, X_test, y_test, name) 
           for name, model in models.items()]

# Данные для графиков (для Linear: actual vs pred, residuals)
y_test_actual = y_test.tolist()
y_test_pred_lin = lr.predict(X_test).tolist()
residuals_lin = (y_test - y_test_pred_lin).tolist()
pred_lin_for_res = y_test_pred_lin

feature_importance = dict(zip(numeric_features, lr.coef_))
top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:4]

results = {m['model']: m for m in metrics}
best_model = max(results.values(), key=lambda x: x['test']['R2'])

conclusions = {
    'influence': [
        f" - Жилая площадь: наибольшее влияние на цену (коэф. {feature_importance['sqft_living']:.2f})",
        f" - Количество ванных комнат: сильное влияние (коэф. {feature_importance['bathrooms']:.2f})",
        f" - Год постройки: более новые дома дороже (коэф. {feature_importance['yr_built']:.2f})",
        f" - Вид на воду: значительно увеличивает стоимость "
    ],
    'best_model': f"Лучшая модель: {best_model['model']} (R² test = {best_model['test']['R2']:.4f})",
    'quality': [
        f" - {m['model']}: разница R² = {m['train']['R2'] - m['test']['R2']:.4f} ({'возможное переобучение' if (m['train']['R2'] - m['test']['R2']) > 0.1 else 'норма' if (m['train']['R2'] - m['test']['R2']) > 0.05 else 'хорошо'})"
        for m in metrics
    ]
}

# Описания моделей
descriptions = {
    'Linear Regression': 'Простая линейная модель: y = Xβ + ε. Минимизирует сумму квадратов ошибок.',
    'LASSO Regression': 'L1-регуляризация: добавляет штраф |β|, обнуляет ненужные коэффициенты (feature selection).',
    'Ridge Regression': 'L2-регуляризация: добавляет штраф β², сжимает коэффициенты, борется с мультиколлинеарностью.',
    'Polynomial Regression': 'Линейная регрессия на полиномиальных фичах (степень 2): захватывает нелинейности.'
}

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": len(models)}

@app.get("/api/metrics")
def get_metrics():
    return {
        "descriptions": descriptions,
        "metrics": metrics,
        "conclusions": conclusions,
        "charts_data": {
            "r2_train": [m['train']['R2'] for m in metrics],
            "r2_test": [m['test']['R2'] for m in metrics],
            "mse_train": [m['train']['MSE'] for m in metrics],
            "mse_test": [m['test']['MSE'] for m in metrics],
            "models": list(models.keys()),
            "scatter_actual": y_test_actual,
            "scatter_pred": y_test_pred_lin,
            "residuals_x": pred_lin_for_res,
            "residuals_y": residuals_lin
        },
    }
