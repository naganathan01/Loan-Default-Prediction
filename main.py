import os
import sys
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "data/raw/Dataset.csv",
        "data/raw/Data_Dictionary.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        
        print(f"\nPlease ensure your data files are in the correct location:")
        print(f"Current working directory: {os.getcwd()}")
        
        # Look for CSV files in current directory
        try:
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                print(f"\n🔍 Found CSV files in current directory:")
                for csv_file in csv_files:
                    print(f"   - {csv_file}")
                    if 'dataset' in csv_file.lower() or 'loan' in csv_file.lower():
                        print(f"     👆 This might be your dataset! Copy it to: data/raw/Dataset.csv")
        except Exception as e:
            print(f"Error listing files: {e}")
        
        return False
    
    return True

def main():
    """Main execution pipeline"""
    
    print("=== LOAN DEFAULT PREDICTION PIPELINE ===")
    
    # Create directories (NOT files!) - Fixed the issue here
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True) 
    os.makedirs("data/models", exist_ok=True)
    
    print("✅ Created necessary directories")
    
    # Check if data files exist
    if not check_data_files():
        print("\n❌ Setup incomplete. Please fix the data file locations and try again.")
        print("\nTo fix this:")
        print("1. Copy your dataset CSV file to: data/raw/Dataset.csv")
        print("2. Copy Data_Dictionary.csv to: data/raw/Data_Dictionary.csv")
        sys.exit(1)
    
    try:
        # Step 1: Data Preprocessing
        print("\n1. Data Preprocessing...")
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        print(f"   ✅ Loaded dataset with shape: {df.shape}")
        
        df_processed = preprocessor.preprocess_data(df)
        print(f"   ✅ Processed data shape: {df_processed.shape}")
        
        # Step 2: Feature Engineering
        print("\n2. Feature Engineering...")
        fe = FeatureEngineer()
        df_features = fe.create_features(df_processed)
        df_encoded = fe.encode_categorical_features(df_features, 'Default')
        print(f"   ✅ Final feature set shape: {df_encoded.shape}")
        
        # Step 3: Model Training
        print("\n3. Model Training...")
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df_encoded)
        print(f"   ✅ Training set: {X_train.shape}, Test set: {X_test.shape}")
        
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
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("\nFor detailed debugging, run: python debug_data_issues.py")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()