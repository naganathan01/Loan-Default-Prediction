import os
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

def main():
    """Main execution pipeline"""
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    print("=== LOAN DEFAULT PREDICTION PIPELINE ===")
    
    # Step 1: Data Preprocessing
    print("\n1. Data Preprocessing...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df_processed = preprocessor.preprocess_data(df)
    
    # Step 2: Feature Engineering
    print("\n2. Feature Engineering...")
    fe = FeatureEngineer()
    df_features = fe.create_features(df_processed)
    df_encoded = fe.encode_categorical_features(df_features, 'Default')
    
    # Step 3: Model Training
    print("\n3. Model Training...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_encoded)
    X_train_balanced, y_train_balanced = trainer.handle_imbalanced_data(X_train, y_train)
    X_train_scaled, X_test_scaled = fe.scale_features(X_train_balanced, X_test)
    
    best_model, model_name = trainer.train_models(X_train_scaled, y_train_balanced, X_test_scaled, y_test)
    model_path = trainer.save_model(best_model, model_name)
    
    # Step 4: Model Evaluation
    print("\n4. Model Evaluation...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(best_model, X_test_scaled, y_test)
    
    # Generate plots
    evaluator.plot_confusion_matrix(best_model, X_test_scaled, y_test)
    evaluator.plot_roc_curve(best_model, X_test_scaled, y_test)
    feature_importance = evaluator.plot_feature_importance(best_model, X_test_scaled.columns)
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"Best Model: {model_name}")
    print(f"Model saved at: {model_path}")
    print(f"AUC-ROC Score: {metrics['AUC-ROC']:.4f}")

if __name__ == "__main__":
    main()