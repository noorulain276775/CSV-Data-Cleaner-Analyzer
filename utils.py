#!/usr/bin/env python3
"""
Utility functions for CSV Data Cleaner Analyzer
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
import re

def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect and suggest optimal data types for columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping column names to suggested data types
    """
    type_suggestions = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it's actually numeric
            if df[col].str.match(r'^\d+\.?\d*$').all():
                type_suggestions[col] = 'float64'
            elif df[col].str.match(r'^\d+$').all():
                type_suggestions[col] = 'int64'
            elif df[col].str.match(r'^\d{4}-\d{2}-\d{2}').all():
                type_suggestions[col] = 'datetime64[ns]'
            else:
                type_suggestions[col] = 'category'
        else:
            type_suggestions[col] = str(df[col].dtype)
            
    return type_suggestions

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing special characters and converting to snake_case
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    cleaned_df = df.copy()
    
    def clean_name(name):
        # Remove special characters and replace spaces with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', str(name))
        cleaned = re.sub(r'\s+', '_', cleaned.strip())
        return cleaned.lower()
    
    cleaned_df.columns = [clean_name(col) for col in cleaned_df.columns]
    return cleaned_df

def find_duplicates(df: pd.DataFrame, subset: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Find duplicate rows in the DataFrame
    
    Args:
        df: Input DataFrame
        subset: Column(s) to consider for duplicates
        
    Returns:
        DataFrame containing duplicate rows
    """
    return df[df.duplicated(subset=subset, keep=False)]

def remove_duplicates(df: pd.DataFrame, subset: Union[str, List[str]] = None, 
                     keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame
    
    Args:
        df: Input DataFrame
        subset: Column(s) to consider for duplicates
        keep: Which duplicates to keep ('first', 'last', False)
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', 
                       columns: List[str] = None) -> pd.DataFrame:
    """
    Fill missing values using various strategies
    
    Args:
        df: Input DataFrame
        strategy: Strategy to use ('mean', 'median', 'mode', 'forward', 'backward')
        columns: Specific columns to fill (None for all numeric columns)
        
    Returns:
        DataFrame with filled missing values
    """
    filled_df = df.copy()
    
    if columns is None:
        columns = filled_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in filled_df.columns:
            if strategy == 'mean':
                filled_df[col].fillna(filled_df[col].mean(), inplace=True)
            elif strategy == 'median':
                filled_df[col].fillna(filled_df[col].median(), inplace=True)
            elif strategy == 'mode':
                filled_df[col].fillna(filled_df[col].mode()[0], inplace=True)
            elif strategy == 'forward':
                filled_df[col].fillna(method='ffill', inplace=True)
            elif strategy == 'backward':
                filled_df[col].fillna(method='bfill', inplace=True)
                
    return filled_df

def detect_outliers(df: pd.DataFrame, columns: List[str] = None, 
                   method: str = 'iqr', threshold: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect outliers in numeric columns
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (None for all numeric columns)
        method: Method to use ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary mapping column names to outlier indices
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_indices = df[z_scores > threshold].index.tolist()
            else:
                outlier_indices = []
                
            outliers[col] = outlier_indices
            
    return outliers

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a numeric column
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        DataFrame with normalized column
    """
    normalized_df = df.copy()
    
    if column not in normalized_df.columns:
        return normalized_df
        
    if method == 'minmax':
        min_val = normalized_df[column].min()
        max_val = normalized_df[column].max()
        normalized_df[f'{column}_normalized'] = (normalized_df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = normalized_df[column].mean()
        std_val = normalized_df[column].std()
        normalized_df[f'{column}_normalized'] = (normalized_df[column] - mean_val) / std_val
    elif method == 'robust':
        median_val = normalized_df[column].median()
        mad_val = np.median(np.abs(normalized_df[column] - median_val))
        normalized_df[f'{column}_normalized'] = (normalized_df[column] - median_val) / mad_val
        
    return normalized_df

def generate_data_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing data quality metrics
    """
    report = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df) * 100),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    # Add statistics for numeric columns
    if report['numeric_columns']:
        report['numeric_stats'] = df[report['numeric_columns']].describe().to_dict()
        
    # Add value counts for categorical columns
    if report['categorical_columns']:
        report['categorical_stats'] = {}
        for col in report['categorical_columns']:
            report['categorical_stats'][col] = df[col].value_counts().head(10).to_dict()
            
    return report

def export_data(df: pd.DataFrame, file_path: str, format: str = 'csv') -> bool:
    """
    Export DataFrame to various formats
    
    Args:
        df: DataFrame to export
        file_path: Output file path
        format: Export format ('csv', 'excel', 'json', 'parquet')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return True
    except Exception as e:
        print(f"Export failed: {str(e)}")
        return False
