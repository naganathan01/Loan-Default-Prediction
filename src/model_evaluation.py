import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'AUC-ROC': roc_auc_score(y_test, y_prob),
            'AUC-PR': average_precision_score(y_test, y_prob),
            'Classification Report': classification_report(y_test, y_pred)
        }
        
        print("=== MODEL EVALUATION RESULTS ===")
        print(f"AUC-ROC: {self.metrics['AUC-ROC']:.4f}")
        print(f"AUC-PR: {self.metrics['AUC-PR']:.4f}")
        print("\nClassification Report:")
        print(self.metrics['Classification Report'])
        
        return self.metrics
    
    def plot_confusion_matrix(self, model, X_test, y_test):
        """Plot confusion matrix"""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    def plot_roc_curve(self, model, X_test, y_test):
        """Plot ROC curve"""
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(20), y='feature', x='importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.show()
            
            return feature_importance
    
    def explain_with_shap(self, model, X_test):
        """Generate SHAP explanations"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:100])  # Sample for speed
            
            # Summary plot
            shap.summary_plot(shap_values, X_test.iloc[:100])
            
            return shap_values
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None

if __name__ == "__main__":
    import joblib
    from feature_engineering import FeatureEngineer
    
    # Load test data and model
    df = pd.read_csv("data/processed/processed_data.csv")
    fe = FeatureEngineer()
    df = fe.create_features(df)
    
    # Load saved model
    model = joblib.load("data/models/best_model_XGBoost.joblib")
    
    # Evaluate
    evaluator = ModelEvaluator()
    # Add your evaluation code here