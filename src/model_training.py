import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import yaml

class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def prepare_data(self, df):
        """Prepare data for training"""
        target_col = self.config['model']['target_column']
        
        # Separate features and target
        X = df.drop([target_col, 'ID'], axis=1, errors='ignore')
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalanced_data(self, X_train, y_train):
        """Handle imbalanced dataset"""
        # Use SMOTE + Random Undersampling
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy=0.5, random_state=42),
            random_state=42
        )
        X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
        
        print(f"Original distribution: {np.bincount(y_train)}")
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and return the best one"""
        
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(
                class_weight='balanced', 
                n_estimators=100, 
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                **self.config['model']['models']['xgboost'],
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"model_{name}"):
                print(f"\nTraining {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=self.config['evaluation']['cv_folds'], 
                    scoring='roc_auc'
                )
                
                mean_cv_score = cv_scores.mean()
                print(f"{name} CV AUC: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Log parameters and metrics
                mlflow.log_params(model.get_params())
                mlflow.log_metric("cv_auc_mean", mean_cv_score)
                mlflow.log_metric("cv_auc_std", cv_scores.std())
                
                # Save model
                mlflow.sklearn.log_model(model, f"model_{name}")
                
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_model = model
                    best_model_name = name
        
        print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")
        return best_model, best_model_name
    
    def save_model(self, model, model_name):
        """Save the trained model"""
        model_path = f"data/models/best_model_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return model_path

if __name__ == "__main__":
    # Load processed data
    from feature_engineering import FeatureEngineer
    
    df = pd.read_csv("data/processed/processed_data.csv")
    
    # Feature engineering
    fe = FeatureEngineer()
    df = fe.create_features(df)
    df = fe.encode_categorical_features(df, 'Default')
    
    # Train model
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Handle imbalanced data
    X_train_balanced, y_train_balanced = trainer.handle_imbalanced_data(X_train, y_train)
    
    # Scale features
    X_train_scaled, X_test_scaled = fe.scale_features(X_train_balanced, X_test)
    
    # Train models
    best_model, model_name = trainer.train_models(X_train_scaled, y_train_balanced, X_test_scaled, y_test)
    
    # Save model
    model_path = trainer.save_model(best_model, model_name)