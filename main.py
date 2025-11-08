from typing import Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1 Загрузка данных
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# 2 Статистика по датасету
def dataset_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.describe(include='all').T
    stats['missing_count'] = df.isnull().sum()
    stats['missing_pct'] = df.isnull().mean().round(4)
    stats['unique_values'] = df.nunique()
    stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                   'missing_count', 'missing_pct', 'unique_values']]
    return stats

# 3 Визуализация данных
def plot_basic_stats(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(num_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.1)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f"Распределение: {col}", fontsize=12)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i])
        axes[i].set_title(f"Boxplot: {col}", fontsize=12)
    plt.tight_layout()
    plt.show()

    corr = df[num_cols].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.show()


# 4 Предобработка
def preprocess_dataset(df: pd.DataFrame, numerical_features: list,
                       categorical_features: list = None, target_column: str = None) -> pd.DataFrame:
    df = df.copy()
    categorical_features = categorical_features or []

    # Удаляем пустые строки
    df = df.dropna().reset_index(drop=True)

    # Кодируем категориальные признаки в 0/1
    if categorical_features:
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Масштабирование числовых
    scaler = StandardScaler()
    features_to_scale = [col for col in numerical_features if col != target_column]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Приведение всех данных к float
    df = df.astype(float)
    return df


# 5 Разделение
def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 6 Линейная регрессия (градиентный спуск)
def linear_regression(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.5, n_steps: int = 1500):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(n_steps):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - ss_residual / ss_total

def evaluate_model(model: tuple, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.dot(X, model[0]) + model[1]
    return r2_score_manual(y, y_pred)

# 7 Построение моделей
def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    model1 = linear_regression(X_train, y_train)
    r2_1 = evaluate_model(model1, X_test, y_test)

    study_features = ['Hours Studied', 'Sample Question Papers Practiced']
    model2 = linear_regression(X_train[study_features], y_train)
    r2_2 = evaluate_model(model2, X_test[study_features], y_test)

    socio_features = [col for col in X_train.columns if col not in study_features]
    model3 = linear_regression(X_train[socio_features], y_train)
    r2_3 = evaluate_model(model3, X_test[socio_features], y_test)

    return {"model_1": r2_1, "model_2": r2_2, "model_3": r2_3}

# 8 Синтетический признак
def build_model_with_synthetic_feature(X_train, X_test, y_train, y_test):
    X_train_syn = X_train.copy()
    X_test_syn = X_test.copy()
    X_train_syn["Study Efficiency"] = X_train_syn["Hours Studied"] * X_train_syn["Sample Question Papers Practiced"]
    X_test_syn["Study Efficiency"] = X_test_syn["Hours Studied"] * X_test_syn["Sample Question Papers Practiced"]

    model = linear_regression(X_train_syn, y_train)
    y_pred = np.dot(X_test_syn, model[0]) + model[1]
    return r2_score_manual(y_test, y_pred)

def main(path="data/Student_Performance.csv"):
    df = load_data(path)
    print(dataset_statistics(df))
    plot_basic_stats(df)

    numerical_features = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
    categorical_features = ['Extracurricular Activities']
    target_column = 'Performance Index'

    df_preprocessed = preprocess_dataset(df, numerical_features, categorical_features, target_column)
    X_train, X_test, y_train, y_test = split_dataset(df_preprocessed, target_column)

    r2_results = build_and_evaluate_models(X_train, X_test, y_train, y_test)
    r2_synthetic = build_model_with_synthetic_feature(X_train, X_test, y_train, y_test)

    print("R² моделей:", r2_results)
    print("R² модели с синтетическим признаком:", r2_synthetic)

if __name__ == "__main__":
    main()

