from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np
import math
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def clean_json_data(obj):
    """Рекурсивно очищает данные от некорректных float значений"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0  # или None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    else:
        return obj
    
app = FastAPI()

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка данных о домах
try:
    df = pd.read_csv('kc_house_data.csv', quotechar='"')  
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S') 
    numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                        'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    X = df[numeric_features].fillna(0)
    y = df['price']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

    # Обучение моделей для домов
    models = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    # 2. LASSO 
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train, y_train)
    models['LASSO Regression'] = lasso

    # 3. Ridge 
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models['Ridge Regression'] = ridge

    # 4. Polynomial 
    poly = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
    poly.fit(X_train, y_train)  
    models['Polynomial Regression'] = poly

    house_models_loaded = True
except Exception as e:
    print(f"Error loading house data: {e}")
    house_models_loaded = False
    models = {}

# Функция для метрик домов
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

# Загрузка данных о мошенничестве
try:
    # Оптимизированные типы данных
    dtypes = {f'V{i}': np.float32 for i in range(1, 29)}
    dtypes['Time'] = np.float32
    dtypes['Amount'] = np.float32  
    dtypes['Class'] = np.int8
    
    fraud_df = pd.read_csv('creditcard.csv', dtype=dtypes)
    fraud_df = fraud_df.drop('Time', axis=1)  # Удаляем сразу
    
    # Подготовка данных
    X_fraud = fraud_df.drop('Class', axis=1)
    y_fraud = fraud_df['Class']
    
    # Оптимизированный StandardScaler
    scaler = StandardScaler()
    X_fraud_scaled = scaler.fit_transform(X_fraud).astype(np.float32)
    
    # Уменьшить test_size для экономии памяти
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
        X_fraud_scaled, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud  # было 0.3
    )
    

    # Обучение модели логистической регрессии
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_fraud_train, y_fraud_train)

    # Предсказания
    y_fraud_pred = logreg.predict(X_fraud_test)
    y_fraud_pred_proba = logreg.predict_proba(X_fraud_test)[:, 1]

    # Метрики без кросс-валидации
    precision = precision_score(y_fraud_test, y_fraud_pred)
    recall = recall_score(y_fraud_test, y_fraud_pred)
    f1 = f1_score(y_fraud_test, y_fraud_pred)
    roc_auc = roc_auc_score(y_fraud_test, y_fraud_pred_proba)

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_fraud_test, y_fraud_pred_proba)

    # Кросс-валидация
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_precision = cross_val_score(logreg, X_fraud_scaled, y_fraud, cv=cv, scoring='precision')
    cv_recall = cross_val_score(logreg, X_fraud_scaled, y_fraud, cv=cv, scoring='recall')
    cv_f1 = cross_val_score(logreg, X_fraud_scaled, y_fraud, cv=cv, scoring='f1')
    cv_roc_auc = cross_val_score(logreg, X_fraud_scaled, y_fraud, cv=cv, scoring='roc_auc')

    # Статистика по классам
    class_distribution = fraud_df['Class'].value_counts().to_dict()
    fraud_percentage = (class_distribution[1] / len(fraud_df)) * 100

    fraud_analysis_loaded = True
except Exception as e:
    print(f"Error loading fraud data: {e}")
    fraud_analysis_loaded = False

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "house_models_loaded": house_models_loaded,
        "fraud_analysis_loaded": fraud_analysis_loaded
    }

@app.get("/api/metrics")
def get_metrics():
    if not house_models_loaded:
        return {"error": "House data not loaded"}
    
    # Вычисление метрик для домов
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

@app.get("/api/fraud-analysis")
def get_fraud_analysis():
    if not fraud_analysis_loaded:
        return {"error": "Fraud data not loaded"}
    
    result = {
        "dataset_info": {
            "total_samples": len(fraud_df),
            "normal_transactions": class_distribution[0],
            "fraud_transactions": class_distribution[1],
            "fraud_percentage": round(fraud_percentage, 4),
            "features_used": X_fraud.shape[1],
            "feature_names": X_fraud.columns.tolist()
        },
        "model_info": {
            "model_type": "Logistic Regression",
            "standardization": "StandardScaler applied",
            "test_size": 0.3,
            "random_state": 42
        },
        "metrics_without_cv": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4)
        },
        "metrics_with_cv": {
            "precision": {
                "mean": round(cv_precision.mean(), 4),
                "std": round(cv_precision.std(), 4),
                "values": cv_precision.tolist()
            },
            "recall": {
                "mean": round(cv_recall.mean(), 4),
                "std": round(cv_recall.std(), 4),
                "values": cv_recall.tolist()
            },
            "f1_score": {
                "mean": round(cv_f1.mean(), 4),
                "std": round(cv_f1.std(), 4),
                "values": cv_f1.tolist()
            },
            "roc_auc": {
                "mean": round(cv_roc_auc.mean(), 4),
                "std": round(cv_roc_auc.std(), 4),
                "values": cv_roc_auc.tolist()
            }
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        },
        "feature_importance": {
            "top_features": dict(zip(X_fraud.columns, logreg.coef_[0])),
            "most_important": sorted(
                [(feature, abs(weight)) for feature, weight in zip(X_fraud.columns, logreg.coef_[0])],
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    }
    

@app.get("/api/heart-attack-analysis")
def get_heart_attack_analysis():
    try:
        # Загрузка данных
        heart_df = pd.read_csv('heart_attack_prediction_dataset.csv')
        
        # Определяем числовые колонки динамически
        numeric_columns = []
        for col in heart_df.columns:
            if col != 'Patient ID' and col != 'Heart Attack Risk':
                try:
                    # Пробуем преобразовать в число
                    pd.to_numeric(heart_df[col])
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    # Если не получается, пропускаем (это категориальная колонка)
                    continue
        
        # Добавляем целевую переменную
        target_column = 'Heart Attack Risk'
        
        print(f"Найдены числовые колонки: {numeric_columns}")
        print(f"Целевая переменная: {target_column}")
        
        # Вычисляем корреляцию с целевой переменной
        correlations = heart_df[numeric_columns + [target_column]].corr()[target_column].abs()
        
        # Удаляем признаки с низкой корреляцией (< 0.05)
        low_corr_features = correlations[correlations < 0.05].index.tolist()
        low_corr_features = [f for f in low_corr_features if f != target_column]
        
        print(f"Признаки с низкой корреляцией: {low_corr_features}")
        
        # Также удаляем сильно коррелирующие между собой признаки
        corr_matrix = heart_df[numeric_columns].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
        
        features_to_remove = list(set(low_corr_features + high_corr_features))
        features_kept = [f for f in numeric_columns if f not in features_to_remove]
        
        print(f"Удаленные признаки: {features_to_remove}")
        print(f"Оставшиеся признаки: {features_kept}")
        
        # Подготовка данных
        X = heart_df[features_kept].copy()
        y = heart_df[target_column]
        
        # Кодирование категориальных признаков
        categorical_columns = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere', 'Blood Pressure']
        for col in categorical_columns:
            if col in heart_df.columns:
                le = LabelEncoder()
                # Заменяем NaN на строку 'Unknown' перед кодированием
                heart_df[col] = heart_df[col].fillna('Unknown')
                heart_df[f'{col}_encoded'] = le.fit_transform(heart_df[col].astype(str))
                X[f'{col}_encoded'] = heart_df[f'{col}_encoded']
        
        print(f"Признаки после кодирования: {X.columns.tolist()}")
        
        # Нормализация данных
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Размер train: {X_train.shape}, test: {X_test.shape}")
        
        # Обучение моделей
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        results = {}
        feature_names = X.columns.tolist()
        
        for name, model in models.items():
            print(f"Обучение модели: {name}")
            
            # Обучение
            model.fit(X_train, y_train)
            
            # Предсказания
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Метрики
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            results[name] = {
                'train_accuracy': round(train_accuracy, 4),
                'test_accuracy': round(test_accuracy, 4),
                'model_type': name,
                'feature_importance': get_feature_importance(model, feature_names, name)
            }
            
            print(f"{name} - Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
        
        # Статистика по классам
        class_distribution = heart_df[target_column].value_counts().to_dict()
        positive_percentage = (class_distribution.get(1, 0) / len(heart_df)) * 100
        
        return {
            "preprocessing_info": {
                "original_features": len(numeric_columns + categorical_columns),
                "features_after_selection": len(feature_names),
                "removed_features": features_to_remove,
                "kept_features": feature_names,
                "scaler_used": "RobustScaler",
                "test_size": 0.2
            },
            "models_results": results,
            "dataset_info": {
                "total_samples": len(heart_df),
                "positive_cases": class_distribution.get(1, 0),
                "negative_cases": class_distribution.get(0, 0),
                "positive_percentage": round(positive_percentage, 2)
            }
        }
        
    except Exception as e:
        print(f"Error in heart attack analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def get_feature_importance(model, feature_names, model_name):
    """Получает важность признаков в зависимости от типа модели"""
    try:
        if model_name == 'Decision Tree' and hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Нормализуем значения
            max_importance = max(importance_dict.values()) if importance_dict.values() else 1.0
            if max_importance > 0:
                return {k: v / max_importance for k, v in importance_dict.items()}
            return importance_dict
            
        elif model_name == 'Random Forest' and hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Нормализуем значения
            max_importance = max(importance_dict.values()) if importance_dict.values() else 1.0
            if max_importance > 0:
                return {k: v / max_importance for k, v in importance_dict.items()}
            return importance_dict
            
        elif model_name == 'SVM' and hasattr(model, 'coef_'):
            importance_dict = dict(zip(feature_names, np.abs(model.coef_[0])))
            # Нормализуем значения
            max_importance = max(importance_dict.values()) if importance_dict.values() else 1.0
            if max_importance > 0:
                return {k: v / max_importance for k, v in importance_dict.items()}
            return importance_dict
            
        else:
            # Возвращаем равномерное распределение если не можем получить важность
            return {feature: 1.0 / len(feature_names) for feature in feature_names}
            
    except Exception as e:
        print(f"Error getting feature importance for {model_name}: {e}")
        return {feature: 0.0 for feature in feature_names}

    # ОЧИСТКА ДАННЫХ ПЕРЕД ВОЗВРАТОМ
    return clean_json_data(result)

@app.get("/api/customer-clustering")
def get_customer_clustering():
    """
    Эндпоинт для кластеризации клиентов
    """
    try:
        # Загрузка данных
        try:
            customer_df = pd.read_csv('customer_data.csv')
        except FileNotFoundError:
            return {
                "error": "Файл customer_data.csv не найден. Пожалуйста, загрузите файл с данными.",
                "file_required": True
            }
        
        print(f"Загружено {len(customer_df)} записей, {len(customer_df.columns)} признаков")
        
        # 1. Обучение моделей кластеризации на исходных данных
        
        # Выбираем только числовые признаки для кластеризации
        numeric_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {
                "error": "Недостаточно числовых признаков для кластеризации",
                "available_columns": customer_df.columns.tolist()
            }
        
        X = customer_df[numeric_cols].values
        
        # 2. Метод локтя для определения оптимального числа кластеров (K-means)
        wcss = []
        silhouette_scores = []
        k_range = range(2, min(11, len(customer_df)))  # Ограничиваем максимальное k
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                wcss.append(float(kmeans.inertia_))
                
                if k > 1:
                    silhouette = silhouette_score(X, kmeans.labels_)
                    silhouette_scores.append(float(silhouette))
                else:
                    silhouette_scores.append(0.0)
            except Exception as e:
                print(f"Ошибка при k={k}: {e}")
                wcss.append(0.0)
                silhouette_scores.append(0.0)
        
        # Определение оптимального k
        optimal_k = 3  # Значение по умолчанию
        if len(wcss) > 1 and any(w > 0 for w in wcss):
            try:
                # Простой алгоритм для нахождения "локтя"
                elbow_index = 0
                max_reduction = 0
                for i in range(1, len(wcss)):
                    reduction = wcss[i-1] - wcss[i]
                    if reduction > max_reduction:
                        max_reduction = reduction
                        elbow_index = i
                optimal_k = elbow_index + 2  # +2 потому что k начинается с 2
                optimal_k = min(optimal_k, 6)  # Ограничиваем максимальное значение
            except:
                optimal_k = 3
        
        # 3. Обучение моделей кластеризации
        
        # 3.1 K-Means Clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        # 3.2 Agglomerative Clustering
        agglomerative = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        agglomerative_labels = agglomerative.fit_predict(X)
        
        # 3.3 DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        
        n_clusters_dbscan = len(np.unique(dbscan_labels[dbscan_labels != -1]))
        if n_clusters_dbscan <= 1:  # Если не нашел кластеры
            # Пробуем другие параметры
            dbscan = DBSCAN(eps=1.0, min_samples=3)
            dbscan_labels = dbscan.fit_predict(X)
        
        # 4. Оценка качества кластеризации
        
        def evaluate_clustering(labels, X_data, method_name):
            """Оценка качества кластеризации"""
            unique_labels = np.unique(labels)
            n_valid_clusters = len(unique_labels[unique_labels != -1])
            
            metrics = {
                'n_clusters': int(n_valid_clusters),
                'n_noise': int(np.sum(labels == -1)) if -1 in labels else 0,
                'cluster_sizes': {}
            }
            
            if n_valid_clusters > 1:
                try:
                    # Убираем шумовые точки для вычисления метрик
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        silhouette = silhouette_score(X_data[mask], labels[mask])
                        metrics['silhouette_score'] = float(silhouette)
                    else:
                        metrics['silhouette_score'] = None
                except:
                    metrics['silhouette_score'] = None
                
                try:
                    if np.sum(mask) > 1:
                        calinski = calinski_harabasz_score(X_data[mask], labels[mask])
                        metrics['calinski_harabasz_score'] = float(calinski)
                    else:
                        metrics['calinski_harabasz_score'] = None
                except:
                    metrics['calinski_harabasz_score'] = None
                
                try:
                    if np.sum(mask) > 1:
                        davies = davies_bouldin_score(X_data[mask], labels[mask])
                        metrics['davies_bouldin_score'] = float(davies)
                    else:
                        metrics['davies_bouldin_score'] = None
                except:
                    metrics['davies_bouldin_score'] = None
            else:
                metrics['silhouette_score'] = None
                metrics['calinski_harabasz_score'] = None
                metrics['davies_bouldin_score'] = None
            
            # Статистика по размерам кластеров
            for label in unique_labels:
                count = int(np.sum(labels == label))
                metrics['cluster_sizes'][str(label)] = count
            
            return metrics
        
        # Оценка каждой модели
        kmeans_evaluation = evaluate_clustering(kmeans_labels, X, 'K-Means')
        agglomerative_evaluation = evaluate_clustering(agglomerative_labels, X, 'Agglomerative')
        dbscan_evaluation = evaluate_clustering(dbscan_labels, X, 'DBSCAN')
        
        # 5. Разведочный анализ данных по кластерам
        
        # Добавляем метки кластеров в DataFrame
        analysis_df = customer_df.copy()
        analysis_df['KMeans_Cluster'] = kmeans_labels
        analysis_df['Agglomerative_Cluster'] = agglomerative_labels
        analysis_df['DBSCAN_Cluster'] = dbscan_labels
        
        # Анализ характеристик кластеров для K-Means
        cluster_analysis = {}
        
        for cluster_num in range(optimal_k):
            cluster_data = analysis_df[analysis_df['KMeans_Cluster'] == cluster_num]
            
            if len(cluster_data) > 0:
                # Вычисляем средние значения для числовых признаков
                cluster_means = {}
                for col in numeric_cols[:10]:  # Берем первые 10 признаков для анализа
                    cluster_means[col] = float(cluster_data[col].mean())
                
                cluster_analysis[f'cluster_{cluster_num}'] = {
                    'size': int(len(cluster_data)),
                    'percentage': float(len(cluster_data) / len(analysis_df) * 100),
                    'mean_values': cluster_means
                }
        
        # 6. Подготовка данных для визуализации на фронтенде
        
        # Выбираем 2 наиболее важных признака для 2D визуализации
        # Используем первые 2 числовых признака или с наибольшей дисперсией
        if len(numeric_cols) >= 2:
            # Выбираем признаки с наибольшей дисперсией
            variances = {}
            for col in numeric_cols:
                variances[col] = float(analysis_df[col].var())
            
            sorted_vars = sorted(variances.items(), key=lambda x: x[1], reverse=True)
            top_features = [sorted_vars[0][0], sorted_vars[1][0]] if len(sorted_vars) >= 2 else numeric_cols[:2]
            
            # Данные для 2D визуализации
            viz_data_2d = []
            for idx, row in analysis_df.iterrows():
                viz_data_2d.append({
                    'x': float(row[top_features[0]]),
                    'y': float(row[top_features[1]]),
                    'kmeans_cluster': int(row['KMeans_Cluster']),
                    'agglomerative_cluster': int(row['Agglomerative_Cluster']),
                    'dbscan_cluster': int(row['DBSCAN_Cluster'])
                })
        else:
            viz_data_2d = []
            top_features = numeric_cols if numeric_cols else []
        
        # 7. Подготовка метрик для графиков
        elbow_data = []
        silhouette_data = []
        
        for i, k in enumerate(k_range):
            if i < len(wcss) and i < len(silhouette_scores):
                elbow_data.append({
                    'k': int(k),
                    'wcss': wcss[i]
                })
                silhouette_data.append({
                    'k': int(k),
                    'silhouette': silhouette_scores[i]
                })
        
        # 8. Анализ распределения по кластерам
        cluster_distributions = {}
        
        for method_name, labels, eval_metrics in [
            ('kmeans', kmeans_labels, kmeans_evaluation),
            ('agglomerative', agglomerative_labels, agglomerative_evaluation),
            ('dbscan', dbscan_labels, dbscan_evaluation)
        ]:
            unique_labels, counts = np.unique(labels, return_counts=True)
            distribution = []
            for label, count in zip(unique_labels, counts):
                distribution.append({
                    'label': str(label),
                    'count': int(count),
                    'percentage': float(count / len(labels) * 100)
                })
            
            cluster_distributions[method_name] = {
                'distribution': distribution,
                'metrics': eval_metrics
            }
        
        # 9. Подготовка результата
        result = {
            "dataset_info": {
                "total_samples": len(customer_df),
                "total_features": len(customer_df.columns),
                "numeric_features": numeric_cols,
                "all_features": customer_df.columns.tolist()
            },
            "clustering_results": {
                "optimal_k": optimal_k,
                "elbow_method_data": elbow_data,
                "silhouette_method_data": silhouette_data,
                "models": {
                    "kmeans": {
                        "labels": [int(label) for label in kmeans_labels],
                        "metrics": kmeans_evaluation,
                        "centers": kmeans.cluster_centers_.tolist() if hasattr(kmeans, 'cluster_centers_') else []
                    },
                    "agglomerative": {
                        "labels": [int(label) for label in agglomerative_labels],
                        "metrics": agglomerative_evaluation
                    },
                    "dbscan": {
                        "labels": [int(label) for label in dbscan_labels],
                        "metrics": dbscan_evaluation
                    }
                }
            },
            "visualization_data": {
                "scatter_2d": {
                    "data": viz_data_2d,
                    "features": top_features,
                    "feature_names": top_features
                },
                "cluster_distributions": cluster_distributions
            },
            "cluster_analysis": cluster_analysis,
            "interpretation": {
                "summary": _generate_clustering_summary(
                    kmeans_evaluation, 
                    agglomerative_evaluation, 
                    dbscan_evaluation,
                    optimal_k
                ),
                "recommendations": _generate_cluster_insights(cluster_analysis, customer_df)
            }
        }
        
        return clean_json_data(result)
        
    except Exception as e:
        print(f"Error in customer clustering: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Ошибка при кластеризации: {str(e)}"}

def _generate_clustering_summary(kmeans_eval, agglo_eval, dbscan_eval, optimal_k):
    """Генерирует сводку по результатам кластеризации"""
    summary = []
    
    summary.append(f"Оптимальное количество кластеров: {optimal_k}")
    summary.append(f"K-Means создал {kmeans_eval['n_clusters']} кластеров")
    summary.append(f"Agglomerative создал {agglo_eval['n_clusters']} кластеров")
    
    if dbscan_eval['n_noise'] > 0:
        summary.append(f"DBSCAN обнаружил {dbscan_eval['n_clusters']} кластеров и {dbscan_eval['n_noise']} шумовых точек")
    else:
        summary.append(f"DBSCAN создал {dbscan_eval['n_clusters']} кластеров")
    
    # Определяем лучшую модель по silhouette score
    models_scores = []
    if kmeans_eval.get('silhouette_score'):
        models_scores.append(('K-Means', kmeans_eval['silhouette_score']))
    if agglo_eval.get('silhouette_score'):
        models_scores.append(('Agglomerative', agglo_eval['silhouette_score']))
    if dbscan_eval.get('silhouette_score'):
        models_scores.append(('DBSCAN', dbscan_eval['silhouette_score']))
    
    if models_scores:
        best_model = max(models_scores, key=lambda x: x[1])
        summary.append(f"Лучшая модель: {best_model[0]} (Silhouette Score: {best_model[1]:.3f})")
    
    return summary

def _generate_cluster_insights(cluster_analysis, df):
    """Генерирует инсайты по кластерам"""
    insights = []
    
    for cluster_key, cluster_info in cluster_analysis.items():
        if cluster_info['size'] > 0:
            cluster_num = cluster_key.split('_')[1]
            size = cluster_info['size']
            percentage = cluster_info['percentage']
            means = cluster_info['mean_values']
            
            insight = f"Кластер {cluster_num}: {size} клиентов ({percentage:.1f}%)"
            
            # Анализируем ключевые метрики, если они есть
            key_metrics = ['Income', 'Recency', 'MntWines', 'MntMeatProducts']
            found_metrics = []
            
            for metric in key_metrics:
                if metric in means:
                    value = means[metric]
                    found_metrics.append(f"{metric}: {value:.0f}")
            
            if found_metrics:
                insight += f" | {', '.join(found_metrics)}"
            
            insights.append(insight)
    
    return insights