import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        """Create new features"""
        df = df.copy()
        print("\n=== Creating New Features ===")
        
        # Age in years (convert from days)
        if 'Age_Days' in df.columns:
            # Clean and convert Age_Days to numeric
            df['Age_Days'] = pd.to_numeric(df['Age_Days'], errors='coerce')
            # Handle missing values by filling with median
            df['Age_Days'] = df['Age_Days'].fillna(df['Age_Days'].median())
            # Create Age_Years feature
            df['Age_Years'] = abs(df['Age_Days']) / 365.25
            print("✓ Age_Years created")
        
        # Employment features
        if 'Employed_Days' in df.columns:
            # Clean and convert to numeric
            df['Employed_Days'] = pd.to_numeric(df['Employed_Days'], errors='coerce')
            df['Employed_Days'] = df['Employed_Days'].fillna(0)  # Fill with 0 for unemployed
            
            df['Is_Employed'] = (df['Employed_Days'] > 0).astype(int)
            df['Employment_Years'] = abs(df['Employed_Days']) / 365.25
            print("✓ Employment features created")
        
        # Financial ratios
        if 'Credit_Amount' in df.columns and 'Client_Income' in df.columns:
            # Clean and convert to numeric
            df['Credit_Amount'] = pd.to_numeric(df['Credit_Amount'], errors='coerce')
            df['Client_Income'] = pd.to_numeric(df['Client_Income'], errors='coerce')
            
            # Fill missing values with median
            df['Credit_Amount'] = df['Credit_Amount'].fillna(df['Credit_Amount'].median())
            df['Client_Income'] = df['Client_Income'].fillna(df['Client_Income'].median())
            
            df['Debt_to_Income'] = df['Credit_Amount'] / (df['Client_Income'] + 1)
            print("✓ Debt_to_Income ratio created")
            
        if 'Loan_Annuity' in df.columns and 'Client_Income' in df.columns:
            # Clean and convert to numeric
            df['Loan_Annuity'] = pd.to_numeric(df['Loan_Annuity'], errors='coerce')
            df['Loan_Annuity'] = df['Loan_Annuity'].fillna(df['Loan_Annuity'].median())
            
            df['Annuity_to_Income'] = df['Loan_Annuity'] / (df['Client_Income'] + 1)
            print("✓ Annuity_to_Income ratio created")
            
        if 'Credit_Amount' in df.columns and 'Loan_Annuity' in df.columns:
            df['Credit_to_Annuity'] = df['Credit_Amount'] / (df['Loan_Annuity'] + 1)
            print("✓ Credit_to_Annuity ratio created")
        
        # Risk indicators
        asset_columns = ['Car_Owned', 'Bike_Owned', 'House_Own']
        available_assets = [col for col in asset_columns if col in df.columns]
        if available_assets:
            df['Total_Assets'] = df[available_assets].sum(axis=1)
            print("✓ Total_Assets created")
        
        # Risk score
        risk_components = []
        if 'Active_Loan' in df.columns:
            risk_components.append(df['Active_Loan'])
        if 'Social_Circle_Default' in df.columns:
            risk_components.append(df['Social_Circle_Default'] * 10)
        
        if risk_components:
            df['Risk_Score'] = sum(risk_components) / len(risk_components)
            print("✓ Risk_Score created")
        
        # Family financial burden
        if 'Client_Income' in df.columns and 'Client_Family_Members' in df.columns:
            df['Income_per_Family_Member'] = df['Client_Income'] / (df['Client_Family_Members'] + 1)
            print("✓ Income_per_Family_Member created")
        
        return df
    
    def encode_categorical_features(self, df, target_column):
        """Encode categorical features"""
        print("\n=== Encoding Categorical Features ===")
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        
        for column in categorical_columns:
            print(f"Encoding {column}...")
            
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column].astype(str))
            else:
                df[column] = self.label_encoders[column].transform(df[column].astype(str))
        
        print(f"✓ {len(categorical_columns)} categorical features encoded")
        return df
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        print("\n=== Scaling Features ===")
        
        # Only scale numerical columns
        numerical_features = X_train.select_dtypes(include=[np.number]).columns
        print(f"Scaling {len(numerical_features)} numerical features")
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled

if __name__ == "__main__":
    # Test feature engineering
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
        fe = FeatureEngineer()
        df_with_features = fe.create_features(df)
        print(f"\nFeatures created successfully. New shape: {df_with_features.shape}")
        print(f"New columns: {[col for col in df_with_features.columns if col not in df.columns]}")
    except FileNotFoundError:
        print("Processed data not found. Run main.py first.")