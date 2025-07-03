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
        
        # Age in years (convert from days)
        df['Age_Years'] = abs(df['Age_Days']) / 365.25
        
        # Employment features
        df['Is_Employed'] = (df['Employed_Days'] > 0).astype(int)
        df['Employment_Years'] = abs(df['Employed_Days']) / 365.25
        
        # Financial ratios
        df['Debt_to_Income'] = df['Credit_Amount'] / (df['Client_Income'] + 1)
        df['Annuity_to_Income'] = df['Loan_Annuity'] / (df['Client_Income'] + 1)
        df['Credit_to_Annuity'] = df['Credit_Amount'] / (df['Loan_Annuity'] + 1)
        
        # Risk indicators
        df['Total_Assets'] = df['Car_Owned'] + df['Bike_Owned'] + df['House_Own']
        df['Risk_Score'] = (df['Active_Loan'] + df['Social_Circle_Default'] * 10) / 2
        
        # Family financial burden
        df['Income_per_Family_Member'] = df['Client_Income'] / (df['Client_Family_Members'] + 1)
        
        return df
    
    def encode_categorical_features(self, df, target_column):
        """Encode categorical features"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column].astype(str))
            else:
                df[column] = self.label_encoders[column].transform(df[column].astype(str))
        
        return df
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        numerical_features = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled

if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv("data/processed/processed_data.csv")
    fe = FeatureEngineer()
    df_with_features = fe.create_features(df)
    print(f"Features created. New shape: {df_with_features.shape}")