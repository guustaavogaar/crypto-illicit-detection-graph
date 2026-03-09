import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_prepare_data(file_path):
    print(f"Loading features from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Filter known labels only
    # In the Elliptic dataset (after our preprocessing in data_ingestion.py):
    # 'label' column is strings: 'Illicit', 'Licit', 'Unknown'
    df_labeled = df[df['label'].isin(['Illicit', 'Licit'])].copy()
    
    print(f"Labeled dataset shape (excluding Unknowns): {df_labeled.shape}")
    
    # Map to binary target (0 = Licit, 1 = Illicit)
    df_labeled['target'] = df_labeled['label'].map({'Illicit': 1, 'Licit': 0})
    
    # Separate features and target
    # Time step could be useful, txId is not a predictive feature
    drop_cols = ['txId', 'label', 'target']
    if 'class' in df_labeled.columns:
        drop_cols.append('class')
    X = df_labeled.drop(drop_cols, axis=1)
    y = df_labeled['target']
    
    return X, y, df

def train_evaluate_model(X, y):
    print("\nSplitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training XGBoost Classifier...")
    # Scale pos weight because illicit transactions are a minority class
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    clf.fit(X_train_scaled, y_train)
    
    print("\nEvaluating Model on Test Set:")
    y_pred = clf.predict(X_test_scaled)
    
    print("-" * 50)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)
    
    return clf, scaler, X_train.columns

def plot_feature_importance(clf, feature_names, output_path):
    print(f"\nPlotting Top 20 Feature Importances to {output_path}...")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 20
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title('Top 20 Feature Importances (XGBoost)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print("Plot saved successfully.")

if __name__ == "__main__":
    data_path = "data/processed/graph_features.csv"
    
    # 1. Load data
    X, y, df_full = load_and_prepare_data(data_path)
    
    # 2. Train and evaluate
    model, scaler, feature_names = train_evaluate_model(X, y)
    
    # 3. Visualize feature importance (to see if Graph features were useful)
    plot_feature_importance(model, feature_names, "data/processed/feature_importance.png")
