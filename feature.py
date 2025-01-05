import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_path = '/Users/liang/Library/Fonts/STHeiti Medium.ttc'
font_prop = FontProperties(fname=font_path)




def feature_selection(X_train, y_train):
    """
    Train LightGBM and Logistic Regression models, extract feature importance, and visualize it.

    Args:
    - X_train: DataFrame, training data.
    - y_train: array-like, labels.

    Returns:
    - models: dict, containing the LightGBM and Logistic Regression models.
    - importances: dict, containing the feature importance DataFrames for both models.
    """

    models = {}
    importances = {}

    # 訓練 LightGBM 模型
    print("Training LightGBM...")
    lgb_model = LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_importances = pd.DataFrame({
        'feature': X_train.columns if isinstance(X_train, pd.DataFrame) else range(X_train.shape[1]),
        'importance': lgb_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    models['LightGBM'] = lgb_model
    importances['LightGBM'] = lgb_importances

    # 訓練 Logistic 回歸模型
    print("Training Logistic Regression...")
    logistic_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42)
    logistic_model.fit(X_train, y_train)
    logistic_importances = pd.DataFrame({
        'feature': X_train.columns if isinstance(X_train, pd.DataFrame) else range(X_train.shape[1]),
        'avg_importance': np.mean(np.abs(logistic_model.coef_), axis=0)  # 5 類別的特徵重要度平均
    }).sort_values(by='avg_importance', ascending=False)
    models['Logistic'] = logistic_model
    importances['Logistic'] = logistic_importances
        
    return models, importances


def combine_features(importances, method='intersection', k=20):
    """
    Combine features from LightGBM and Logistic Regression based on importance scores.
    Supports selection by intersection or union.

    Args:
    - importances: dict, A dictionary containing feature importance DataFrames for two models.
                   Keys are model names (e.g., 'LightGBM', 'Logistic'), and values are DataFrames
                   with feature importance scores.
    - method: str, The method to combine features. Options are:
              'intersection' (features present in both models) or 'union' (features from either model).
    - k: int, The number of top features to select from each model.

    Returns:
    - selected_features: list, A list of selected feature names.
    """
    # Get the top-k features for LightGBM
    top_lgb_features = set(importances['LightGBM'].head(k)['feature'])
    
    # Get the top-k features for Logistic Regression
    top_logistic_features = set(importances['Logistic'].head(k)['feature'])

    # Combine features based on the selected method
    if method == 'intersection':
        # Select features that are common to both models
        selected_features = list(top_lgb_features & top_logistic_features)
    elif method == 'union':
        # Select features that are present in either model
        selected_features = list(top_lgb_features | top_logistic_features)
    else:
        # Raise an error if the method is invalid
        raise ValueError("Invalid method. Choose 'intersection' or 'union'.")

    # Print the results
    print(f"Method: {method}")
    print("Selected Features: ", selected_features)
    print("Number of Selected Features: ", len(selected_features))

    return selected_features


def visualize_feature_importance(model_name, importances, importance_col, feature_col, color, title_suffix="Feature Importance"):

    print(f"Visualizing {model_name} {title_suffix}...")
    plt.figure(figsize=(6, 6))
    plt.barh(importances[feature_col], importances[importance_col], color=color)
    plt.gca().invert_yaxis()
    plt.title(f'{model_name} {title_suffix}', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.show()


def perform_pca(train_embeddings, test_embeddings, n_components=None):

    pca = PCA(n_components=n_components)
    
    train_pca = pca.fit_transform(train_embeddings)
    test_pca = pca.transform(test_embeddings)
    
    return train_pca, test_pca, pca


def plot_knee(pca):
    # 獲取累積解釋方差比
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 繪製 Knee Plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title("PCA Knee Plot", fontsize=16)
    plt.xlabel("Number of Components", fontsize=12)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=12)
    plt.grid(True)
    plt.show()


def plot_explained_variance(pca):
    """
    繪製每個主成分對應的解釋方差比例。
    
    Args:
    - pca: 訓練好的 PCA 模型。
    """
    # 獲取每個主成分的解釋方差比例
    explained_variance = pca.explained_variance_ratio_
    
    # 繪製解釋方差比例圖
    plt.figure(figsize=(6, 4))
    plt.bar(range(100), explained_variance[:100], align='center', label="Individual Explained Variance")
    plt.title("PCA Individual Explained Variance", fontsize=16)
    plt.xlabel("Principal Component Index", fontsize=12)
    plt.ylabel("Explained Variance Ratio", fontsize=12)
    plt.grid(True)
    plt.show()
