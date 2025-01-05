# Description: This file contains the model training function that supports SVM, XGBoost, LightGBM, and Logistic Regression.

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score


def train_model(model_name, X_train, y_train, X_test, y_test):
    """
    A general-purpose model training function supporting SVM, XGBoost, LightGBM, and Logistic Regression.
    Returns the trained model and performance metrics.
    
    Args:
    - model_name: str, The name of the model to train ("SVM", "XGB", "LGBM", "Logistic").
    - X_train: array-like, Training feature data.
    - y_train: array-like, Training labels.
    - X_test: array-like, Testing feature data.
    - y_test: array-like, Testing labels.
    
    Returns:
    - model: Trained model object.
    - metrics: dict, A dictionary containing performance metrics (Accuracy, Recall, Precision, F1-Score).
    """
    print(f"Training {model_name}...")

    # Initialize the model based on model_name
    if model_name == "SVM":
        model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    elif model_name == "XGB":
        model = XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)), random_state=42)
    elif model_name == "LGBM":
        model = LGBMClassifier(objective='multiclass', num_class=len(set(y_train)), random_state=42)
    elif model_name == "Logistic":
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42)
    else:
        raise ValueError("Invalid model_name. Choose from 'SVM', 'XGB', 'LGBM', or 'Logistic'.")

    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='macro')
    # recall = recall_score(y_test, y_pred, average='macro') # UAR
    # f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted') 
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print metrics for visibility
    print(f"{model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print("")

    # Return the trained model and performance metrics
    return model, {"Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1-Score": f1}
