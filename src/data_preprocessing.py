import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import yaml

class DataPreprocessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the dataset"""
        df = pd.read_csv(self.config['data']['raw_data_path'])
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values based on missing percentage"""
        missing_thresh = 0.5  # 50% threshold
        
        for column in df.columns:
            missing_pct = df[column].isnull().sum() / len(df)
            
            if missing_pct > missing_thresh:
                # Create missing indicator
                df[f'{column}_missing'] = df[column].isnull().astype(int)
                # Fill with median for numerical, mode for categorical
                if df[column].dtype in ['int64', 'float64']:
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown', inplace=True)
            
            elif missing_pct > 0.15:  # 15-50% missing
                if df[column].dtype in ['int64', 'float64']:
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna('Unknown', inplace=True)
            
            elif missing_pct > 0:  # <15% missing - use KNN
                if df[column].dtype in ['int64', 'float64']:
                    imputer = KNNImputer(n_neighbors=5)
                    df[[column]] = imputer.fit_transform(df[[column]])
                else:
                    df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def handle_outliers(self, df, columns):
        """Handle outliers using IQR method"""
        for column in columns:
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower_bound, upper_bound)
        return df
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        print("✓ Missing values handled")
        
        # Handle outliers for key numerical features
        numerical_features = ['Client_Income', 'Credit_Amount', 'Age_Days', 'Employed_Days']
        df = self.handle_outliers(df, numerical_features)
        print("✓ Outliers handled")
        
        # Save processed data
        processed_path = f"{self.config['data']['processed_data_path']}processed_data.csv"
        df.to_csv(processed_path, index=False)
        print(f"✓ Processed data saved to {processed_path}")
        
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    processed_df = preprocessor.preprocess_data(df)
    print(f"Data shape after preprocessing: {processed_df.shape}")