"""
Crash Data Cleaner Script
=========================
This script cleans crash data by handling missing values:
- Removes columns that are entirely blank
- Fills numerical columns with median
- Fills categorical columns with mode
- Special handling for "victim 1 movement information": fill with "no information"
- Exports cleaned data to xlsx format
"""

import pandas as pd
import numpy as np


def load_data(file_path):
    """Load the Excel crash data file."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_excel(file_path)
    print(f"✓ Loaded data from: {file_path}")
    print(f"✓ Original shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def remove_blank_columns(df):
    """Remove columns that are entirely blank (all NaN values)."""
    print("\n" + "=" * 60)
    print("REMOVING BLANK COLUMNS")
    print("=" * 60)
    
    # Find columns where all values are NaN
    blank_mask = df.isnull().all()
    blank_cols = list(df.columns[blank_mask])
    
    if blank_cols:
        print(f"✓ Found {len(blank_cols)} entirely blank column(s):")
        for col in blank_cols:
            print(f"  - {col}")
        df = df.drop(columns=blank_cols)
        print(f"✓ Removed {len(blank_cols)} blank column(s)")
    else:
        print("✓ No entirely blank columns found")
    
    return df


def fill_numerical_columns(df):
    """Fill missing values in numerical columns with median."""
    print("\n" + "=" * 60)
    print("FILLING NUMERICAL COLUMNS (MEDIAN)")
    print("=" * 60)
    
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"✓ Found {len(numeric_cols)} numerical column(s)")
    
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  - {col}: filled {missing_count} missing values with median: {median_val}")
        else:
            print(f"  - {col}: no missing values")
    
    return df


def fill_categorical_columns(df):
    """Fill missing values in categorical columns with mode."""
    print("\n" + "=" * 60)
    print("FILLING CATEGORICAL COLUMNS (MODE)")
    print("=" * 60)
    
    # Select categorical columns (non-numerical)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"✓ Found {len(categorical_cols)} categorical column(s)")
    
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            # Check if mode exists
            mode_series = df[col].mode()
            if not mode_series.empty:
                mode_val = mode_series[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  - {col}: filled {missing_count} missing values with mode: '{mode_val}'")
            else:
                df[col] = df[col].fillna('No information')
                print(f"  - {col}: filled {missing_count} missing values with 'No information' (no mode found)")
        else:
            print(f"  - {col}: no missing values")
    
    return df


def fill_victim1_movement_info(df):
    """Fill missing values in 'victim 1 movement information' column with 'no information'."""
    print("\n" + "=" * 60)
    print("HANDLING 'VICTIM-1 MOVEMENT'")
    print("=" * 60)
    
    # Check for the column (case-insensitive search)
    victim_col = None
    for col in df.columns:
        if 'Victim-1 Movement' in col.lower():
            victim_col = col
            break
    
    if victim_col:
        missing_count = df[victim_col].isnull().sum()
        if missing_count > 0:
            df[victim_col] = df[victim_col].fillna('No information')
            print(f"✓ Column '{victim_col}': filled {missing_count} missing values with 'No information'")
        else:
            print(f"✓ Column '{victim_col}': no missing values")
    else:
        print("⚠ Column 'victim 1 movement' not found in the dataset")
    
    return df


def check_missing_values(df):
    """Check and display remaining missing values."""
    print("\n" + "=" * 60)
    print("MISSING VALUES SUMMARY")
    print("=" * 60)
    
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing == 0:
        print("✓ No missing values remaining in the dataset!")
    else:
        print(f"⚠ Total missing values remaining: {total_missing}")
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            print("\nColumns with missing values:")
            for col, count in cols_with_missing.items():
                print(f"  - {col}: {count}")
    
    return total_missing


def export_data(df, output_file):
    """Export the cleaned data to Excel format."""
    print("\n" + "=" * 60)
    print("EXPORTING DATA")
    print("=" * 60)
    
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"✓ Exported cleaned data to: {output_file}")
    print(f"✓ Final shape: {df.shape[0]} rows × {df.shape[1]} columns")


def main():
    """Main function to run the data cleaning pipeline."""
    print("\n" + "=" * 60)
    print("  CRASH DATA CLEANER")
    print("  Handling Missing Values in Excel Data")
    print("=" * 60)
    
    # Configuration
    input_file = 'Crash Data AA.xlsx'
    output_file = 'Crash_Data_AA_cleaned.xlsx'
    
    try:
        # Step 1: Load data
        df = load_data(input_file)
        
        # Step 2: Remove entirely blank columns
        df = remove_blank_columns(df)
        
        # Step 3: Fill numerical columns with median
        df = fill_numerical_columns(df)
        
        # Step 4: Fill categorical columns with mode
        df = fill_categorical_columns(df)
        
        # Step 5: Special handling for victim 1 movement information
        df = fill_victim1_movement_info(df)
        
        # Step 6: Check remaining missing values
        check_missing_values(df)
        
        # Step 7: Export cleaned data
        export_data(df, output_file)
        
        # Final summary
        print("\n" + "=" * 60)
        print("  DATA CLEANING COMPLETE!")
        print("=" * 60)
        print(f"\n✓ Input file: {input_file}")
        print(f"✓ Output file: {output_file}")
        print(f"✓ Columns processed: {df.shape[1]}")
        print(f"✓ Rows processed: {df.shape[0]}")
        
        return df
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find file '{input_file}'")
        print("Please ensure the Excel file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    cleaned_df = main()

