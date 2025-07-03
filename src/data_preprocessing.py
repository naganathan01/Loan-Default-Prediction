import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
import yaml
import os

class DataPreprocessor:
    def __init__(self, config_path=r"E:\&%A\master-project-task\Loan-Default-Prediction\config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the dataset"""
        data_path = self.config['data']['raw_data_path']
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Please place Dataset.csv in data/raw/")
        
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values based on missing percentage"""
        missing_thresh_high = 0.5  # 50% threshold
        missing_thresh_medium = 0.15  # 15% threshold
        
        print("Handling missing values...")
        
        for column in df.columns:
            if column in ['ID', 'Default']:  # Skip ID and target
                continue
                
            missing_count = df[column].isnull().sum()
            missing_pct = missing_count / len(df)
            
            if missing_count > 0:
                print(f"  {column}: {missing_count} missing ({missing_pct:.2%})")
            
            if missing_pct > missing_thresh_high:
                # Create missing indicator for high missing columns
                df[f'{column}_missing'] = df[column].isnull().astype(int)
                
                # Fill with median for numerical, mode for categorical
                if df[column].dtype in ['int64', 'float64']:
                    fill_value = df[column].median()
                    df[column].fillna(fill_value, inplace=True)
                else:
                    mode_values = df[column].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                    df[column].fillna(fill_value, inplace=True)
            
            elif missing_pct > missing_thresh_medium:
                # Medium missing: simple imputation
                if df[column].dtype in ['int64', 'float64']:
                    fill_value = df[column].median()
                    df[column].fillna(fill_value, inplace=True)
                else:
                    df[column].fillna('Unknown', inplace=True)
            
            elif missing_pct > 0:
                # Low missing: KNN imputation for numerical, mode for categorical
                if df[column].dtype in ['int64', 'float64']:
                    # Use SimpleImputer as backup if KNN fails
                    try:
                        imputer = KNNImputer(n_neighbors=5)
                        df[[column]] = imputer.fit_transform(df[[column]])
                    except:
                        df[column].fillna(df[column].median(), inplace=True)
                else:
                    mode_values = df[column].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                    df[column].fillna(fill_value, inplace=True)
        
        return df
    
    def handle_outliers(self, df, columns):
        """Handle outliers using IQR method"""
        print("Handling outliers...")
        
        for column in columns:
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_before = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                    df[column] = df[column].clip(lower_bound, upper_bound)
                    
                    if outliers_before > 0:
                        print(f"  {column}: {outliers_before} outliers clipped")
        
        return df
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        print("\n=== Starting Data Preprocessing ===")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        print("✓ Missing values handled")
        
        # Handle outliers for key numerical features
        numerical_features = []
        potential_features = ['Client_Income', 'Credit_Amount', 'Age_Days', 'Employed_Days']
        
        for feat in potential_features:
            if feat in df.columns:
                numerical_features.append(feat)
        
        if numerical_features:
            df = self.handle_outliers(df, numerical_features)
            print("✓ Outliers handled")
        
        # Ensure processed directory exists
        processed_dir = self.config['data']['processed_data_path']
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save processed data
        processed_path = f"{processed_dir}processed_data.csv"
        df.to_csv(processed_path, index=False)
        print(f"✓ Processed data saved to {processed_path}")
        
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    processed_df = preprocessor.preprocess_data(df)
    print(f"\nData shape after preprocessing: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")