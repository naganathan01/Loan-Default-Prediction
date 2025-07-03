#!/usr/bin/env python3
"""
Setup and validation script for Loan Default Prediction project
"""

import os
import sys
import pandas as pd

def check_project_structure():
    """Check if project structure is correct"""
    print("🔍 Checking project structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/models",
        "src",
        "notebooks"
    ]
    
    required_files = [
        "config.yaml",
        "main.py",
        "requirements.txt",
        "src/__init__.py",
        "src/data_preprocessing.py",
        "src/feature_engineering.py",
        "src/model_training.py",
        "src/model_evaluation.py",
        "src/prediction_api.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            os.makedirs(dir_path, exist_ok=True)
    
    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"📁 Created missing directories: {missing_dirs}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ Project structure is correct!")
        return True

def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking data files...")
    
    required_data_files = [
        "data/raw/Dataset.csv",
        "data/raw/Data_Dictionary.csv"
    ]
    
    missing_data = []
    for file_path in required_data_files:
        if not os.path.exists(file_path):
            missing_data.append(file_path)
    
    if missing_data:
        print(f"❌ Missing data files: {missing_data}")
        print("\n📝 Please download and place the following files:")
        for file_path in missing_data:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All data files found!")
        return True

def validate_data():
    """Validate the data files"""
    print("\n🔍 Validating data files...")
    
    try:
        # Check Dataset.csv
        df = pd.read_csv("data/raw/Dataset.csv")
        print(f"✅ Dataset.csv loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check for target column
        if 'Default' in df.columns:
            default_rate = df['Default'].mean()
            print(f"   Default rate: {default_rate:.2%}")
        else:
            print("❌ 'Default' column not found in dataset")
            return False
        
        # Check Data_Dictionary.csv
        dict_df = pd.read_csv("data/raw/Data_Dictionary.csv")
        print(f"✅ Data_Dictionary.csv loaded successfully")
        print(f"   Entries: {len(dict_df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def test_imports():
    """Test required package imports"""
    print("\n📦 Testing package imports...")
    
    packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('xgboost', 'xgb'),
        ('imblearn', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('mlflow', None),
        ('fastapi', None),
        ('joblib', None),
        ('yaml', None)
    ]
    
    failed_imports = []
    
    for package, alias in packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Create sample data
        X = np.random.rand(1000, 10)
        y = np.random.choice([0, 1], 1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Quick test passed! Sample accuracy: {accuracy:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup and validation function"""
    print("=" * 60)
    print("  LOAN DEFAULT PREDICTION - SETUP & VALIDATION")
    print("=" * 60)
    
    all_checks_passed = True
    
    # 1. Check project structure
    if not check_project_structure():
        all_checks_passed = False
    
    # 2. Check data files
    if not check_data_files():
        all_checks_passed = False
    
    # 3. Validate data
    if not validate_data():
        all_checks_passed = False
    
    # 4. Test imports
    if not test_imports():
        all_checks_passed = False
    
    # 5. Run quick test
    if not run_quick_test():
        all_checks_passed = False
    
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print("🎉 SETUP VALIDATION SUCCESSFUL!")
        print("\nNext steps:")
        print("1. Run the main pipeline: python main.py")
        print("2. Start the API server: python src/prediction_api.py")
        print("3. View MLflow experiments: mlflow ui")
        print("4. Explore with Jupyter: jupyter notebook")
    else:
        print("❌ SETUP VALIDATION FAILED!")
        print("\nPlease fix the issues above before proceeding.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()