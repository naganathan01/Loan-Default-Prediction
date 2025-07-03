#!/usr/bin/env python3
"""
Debug script to identify and fix data issues in the dataset
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_data_issues(df):
    """Analyze data issues in the dataset"""
    print("=" * 60)
    print("  DATA ISSUES ANALYSIS")
    print("=" * 60)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check data types
    print("\n📊 DATA TYPES:")
    print(df.dtypes)
    
    # Check for problematic columns
    print("\n🔍 PROBLEMATIC COLUMNS:")
    
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(f"Type: {df[col].dtype}")
        print(f"Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        
        # Check for mixed data types
        if df[col].dtype == 'object':
            unique_types = set(type(x).__name__ for x in df[col].dropna().head(100))
            print(f"Value types: {unique_types}")
            
            # Show sample values
            sample_values = df[col].dropna().head(10).tolist()
            print(f"Sample values: {sample_values}")
        
        # Check for numeric columns with string values
        if col in ['Age_Days', 'Employed_Days', 'Client_Income', 'Credit_Amount', 'Loan_Annuity']:
            print(f"Should be numeric: {col}")
            
            # Try to convert to numeric and see what fails
            try:
                numeric_version = pd.to_numeric(df[col], errors='coerce')
                conversion_issues = df[col][numeric_version.isnull() & df[col].notnull()]
                
                if len(conversion_issues) > 0:
                    print(f"❌ Conversion issues: {len(conversion_issues)} values")
                    print(f"Problematic values: {conversion_issues.head(10).tolist()}")
                else:
                    print("✅ Can be converted to numeric")
            except Exception as e:
                print(f"❌ Error converting: {e}")

def fix_data_issues(df):
    """Fix common data issues"""
    print("\n" + "=" * 60)
    print("  FIXING DATA ISSUES")
    print("=" * 60)
    
    df_fixed = df.copy()
    
    # List of columns that should be numeric
    numeric_columns = [
        'Age_Days', 'Employed_Days', 'Client_Income', 'Credit_Amount', 
        'Loan_Annuity', 'Population_Region_Relative', 'Registration_Days',
        'ID_Days', 'Own_House_Age', 'Client_Family_Members', 'Cleint_City_Rating',
        'Application_Process_Day', 'Application_Process_Hour', 'Score_Source_1',
        'Score_Source_2', 'Social_Circle_Default', 'Phone_Change', 'Credit_Bureau',
        'Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Child_Count'
    ]
    
    for col in numeric_columns:
        if col in df_fixed.columns:
            print(f"\n🔧 Fixing {col}...")
            
            # Store original data type and missing count
            original_dtype = df_fixed[col].dtype
            original_missing = df_fixed[col].isnull().sum()
            
            # Convert to numeric
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
            
            # Check how many new missing values were created
            new_missing = df_fixed[col].isnull().sum()
            conversion_issues = new_missing - original_missing
            
            if conversion_issues > 0:
                print(f"  ⚠️  {conversion_issues} values couldn't be converted (set to NaN)")
            
            # Fill missing values appropriately
            if col in ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own']:
                # Binary columns - fill with 0
                df_fixed[col] = df_fixed[col].fillna(0)
                print(f"  ✅ Filled missing values with 0")
            elif col in ['Child_Count', 'Client_Family_Members']:
                # Count columns - fill with 0 or 1
                fill_value = 1 if 'Family' in col else 0
                df_fixed[col] = df_fixed[col].fillna(fill_value)
                print(f"  ✅ Filled missing values with {fill_value}")
            else:
                # Other numeric columns - fill with median
                if not df_fixed[col].isnull().all():
                    median_val = df_fixed[col].median()
                    df_fixed[col] = df_fixed[col].fillna(median_val)
                    print(f"  ✅ Filled missing values with median: {median_val}")
                else:
                    print(f"  ❌ All values are missing - setting to 0")
                    df_fixed[col] = 0
    
    # Fix categorical columns
    categorical_columns = df_fixed.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col not in ['Default']]
    
    for col in categorical_columns:
        print(f"\n🔧 Fixing categorical column {col}...")
        
        # Fill missing values with 'Unknown'
        missing_count = df_fixed[col].isnull().sum()
        if missing_count > 0:
            df_fixed[col] = df_fixed[col].fillna('Unknown')
            print(f"  ✅ Filled {missing_count} missing values with 'Unknown'")
        
        # Convert to string type
        df_fixed[col] = df_fixed[col].astype(str)
        print(f"  ✅ Converted to string type")
    
    return df_fixed

def validate_fixed_data(df):
    """Validate that data issues are fixed"""
    print("\n" + "=" * 60)
    print("  VALIDATION")
    print("=" * 60)
    
    issues_found = False
    
    # Check for missing values
    missing_summary = df.isnull().sum()
    columns_with_missing = missing_summary[missing_summary > 0]
    
    if len(columns_with_missing) > 0:
        print("❌ Still has missing values:")
        for col, count in columns_with_missing.items():
            print(f"  {col}: {count}")
        issues_found = True
    else:
        print("✅ No missing values found")
    
    # Check data types
    print("\n📊 Final data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Test feature engineering on sample
    print("\n🧪 Testing feature engineering on sample...")
    try:
        sample_df = df.head(100).copy()
        
        # Test Age_Years creation
        if 'Age_Days' in sample_df.columns:
            sample_df['Age_Years'] = abs(sample_df['Age_Days']) / 365.25
            print("✅ Age_Years creation works")
        
        # Test Employment features
        if 'Employed_Days' in sample_df.columns:
            sample_df['Is_Employed'] = (sample_df['Employed_Days'] > 0).astype(int)
            sample_df['Employment_Years'] = abs(sample_df['Employed_Days']) / 365.25
            print("✅ Employment features work")
        
        # Test financial ratios
        if 'Credit_Amount' in sample_df.columns and 'Client_Income' in sample_df.columns:
            sample_df['Debt_to_Income'] = sample_df['Credit_Amount'] / (sample_df['Client_Income'] + 1)
            print("✅ Financial ratios work")
        
        print("✅ All feature engineering tests passed")
        
    except Exception as e:
        print(f"❌ Feature engineering still has issues: {e}")
        issues_found = True
    
    return not issues_found

def main():
    """Main function"""
    print("🔍 DEBUGGING DATA ISSUES")
    
    # Check if processed data exists
    if not os.path.exists("data/processed/processed_data.csv"):
        print("❌ Processed data not found. Run preprocessing first.")
        return
    
    # Load processed data
    print("📂 Loading processed data...")
    df = pd.read_csv("data/processed/processed_data.csv")
    
    # Analyze issues
    analyze_data_issues(df)
    
    # Fix issues
    df_fixed = fix_data_issues(df)
    
    # Validate fixes
    if validate_fixed_data(df_fixed):
        print("\n🎉 All data issues fixed!")
        
        # Save fixed data
        fixed_path = "data/processed/processed_data_fixed.csv"
        df_fixed.to_csv(fixed_path, index=False)
        print(f"✅ Fixed data saved to {fixed_path}")
        
        # Replace original processed data
        df_fixed.to_csv("data/processed/processed_data.csv", index=False)
        print("✅ Original processed data updated")
        
    else:
        print("\n❌ Some issues remain. Please check the output above.")

if __name__ == "__main__":
    main()