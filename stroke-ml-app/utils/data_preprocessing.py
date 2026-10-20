"""
Module de prétraitement des données.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_clean_data(filepath):
    """Charge et nettoie le dataset."""
    df = pd.read_csv(filepath)
    
    # Supprimer la colonne 'id'
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Gérer les valeurs manquantes
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    # Supprimer les lignes avec 'Other' dans gender (rare)
    if 'gender' in df.columns:
        df = df[df['gender'] != 'Other']
    
    return df


def preprocess_data(df):
    """Encode les variables catégorielles."""
    df_processed = df.copy()
    
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
    
    return df_processed, label_encoders


def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Détecte les valeurs aberrantes.
    
    Parameters:
    -----------
    df : DataFrame
        Les données
    columns : list
        Colonnes à analyser (None = toutes les colonnes numériques)
    method : str
        'iqr' ou 'zscore'
    threshold : float
        1.5 pour IQR, 3 pour zscore
        
    Returns:
    --------
    dict : {colonne: indices des outliers}
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers[col] = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = df[z_scores > threshold].index.tolist()
    
    return outliers


def handle_outliers(df, columns=None, method='cap'):
    """
    Traite les valeurs aberrantes.
    
    Parameters:
    -----------
    method : str
        'cap' (winsorization), 'remove', 'transform'
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'cap':
            # Winsorization
            Q1 = df_clean[col].quantile(0.05)
            Q3 = df_clean[col].quantile(0.95)
            df_clean[col] = df_clean[col].clip(lower=Q1, upper=Q3)
        
        elif method == 'remove':
            # Suppression
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    return df_clean


def get_basic_stats(df):
    """Retourne les statistiques de base."""
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist()
    }
    return stats
