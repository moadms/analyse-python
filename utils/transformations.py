"""
Module de transformations des données (normalisation, log, Box-Cox, etc.).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from scipy import stats


class DataTransformer:
    """Classe pour gérer les transformations de données."""
    
    def __init__(self):
        self.transformers = {}
        self.transformation_history = []
    
    def log_transform(self, df, columns=None, shift=1):
        """
        Transformation logarithmique: log(x + shift)
        Utile pour réduire l'asymétrie (skewness)
        """
        df_transformed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if (df_transformed[col] >= 0).all():
                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                self.transformation_history.append(f'{col} -> log transform')
        
        return df_transformed
    
    def sqrt_transform(self, df, columns=None):
        """
        Transformation racine carrée: sqrt(x)
        Moins agressive que log
        """
        df_transformed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if (df_transformed[col] >= 0).all():
                df_transformed[f'{col}_sqrt'] = np.sqrt(df_transformed[col])
                self.transformation_history.append(f'{col} -> sqrt transform')
        
        return df_transformed
    
    def box_cox_transform(self, df, columns=None):
        """
        Transformation Box-Cox (nécessite x > 0)
        Trouve automatiquement le meilleur lambda
        """
        df_transformed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if (df_transformed[col] > 0).all():
                try:
                    df_transformed[f'{col}_boxcox'], lambda_param = stats.boxcox(df_transformed[col])
                    self.transformation_history.append(f'{col} -> Box-Cox (λ={lambda_param:.2f})')
                except:
                    pass
        
        return df_transformed
    
    def yeo_johnson_transform(self, df, columns=None):
        """
        Transformation Yeo-Johnson (fonctionne avec valeurs négatives)
        Alternative à Box-Cox
        """
        df_transformed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        pt = PowerTransformer(method='yeo-johnson')
        
        for col in columns:
            try:
                df_transformed[f'{col}_yj'] = pt.fit_transform(df_transformed[[col]])
                self.transformation_history.append(f'{col} -> Yeo-Johnson')
            except:
                pass
        
        return df_transformed
    
    def quantile_transform(self, df, columns=None, output_distribution='normal'):
        """
        Transformation quantile: transforme en distribution normale ou uniforme
        Robuste aux outliers
        """
        df_transformed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        qt = QuantileTransformer(output_distribution=output_distribution, random_state=42)
        
        for col in columns:
            try:
                df_transformed[f'{col}_quantile'] = qt.fit_transform(df_transformed[[col]])
                self.transformation_history.append(f'{col} -> Quantile ({output_distribution})')
            except:
                pass
        
        return df_transformed
    
    def standard_scale(self, df, columns=None):
        """
        Standardisation: (x - mean) / std
        Mean=0, Std=1
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.transformers['standard'] = scaler
        
        return df_scaled
    
    def minmax_scale(self, df, columns=None, feature_range=(0, 1)):
        """
        Normalisation Min-Max: (x - min) / (max - min)
        Ramène dans [0, 1]
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        scaler = MinMaxScaler(feature_range=feature_range)
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.transformers['minmax'] = scaler
        
        return df_scaled
    
    def robust_scale(self, df, columns=None):
        """
        Robust Scaler: utilise médiane et IQR
        Robuste aux outliers
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        scaler = RobustScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.transformers['robust'] = scaler
        
        return df_scaled
    
    def calculate_skewness(self, df, columns=None):
        """Calcule l'asymétrie (skewness) de chaque colonne."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        skewness = {}
        for col in columns:
            skewness[col] = df[col].skew()
        
        return pd.DataFrame.from_dict(skewness, orient='index', columns=['Skewness'])
    
    def suggest_transformations(self, df, skewness_threshold=0.5):
        """Suggère des transformations basées sur le skewness."""
        skewness = self.calculate_skewness(df)
        suggestions = {}
        
        for col, skew_val in skewness['Skewness'].items():
            if abs(skew_val) < skewness_threshold:
                suggestions[col] = "Aucune transformation nécessaire"
            elif skew_val > 1:
                suggestions[col] = "Log ou Box-Cox (asymétrie droite forte)"
            elif skew_val > skewness_threshold:
                suggestions[col] = "Sqrt ou Yeo-Johnson (asymétrie droite modérée)"
            elif skew_val < -1:
                suggestions[col] = "Transformation inverse (asymétrie gauche forte)"
            else:
                suggestions[col] = "Yeo-Johnson (asymétrie gauche modérée)"
        
        return suggestions


def auto_transform(df, columns=None, method='auto'):
    """
    Transforme automatiquement les colonnes asymétriques.
    
    Parameters:
    -----------
    method : str
        'auto', 'log', 'sqrt', 'boxcox', 'yeo-johnson', 'quantile'
    """
    transformer = DataTransformer()
    
    if method == 'auto':
        skewness = transformer.calculate_skewness(df, columns)
        df_transformed = df.copy()
        
        for col in skewness.index:
            skew_val = skewness.loc[col, 'Skewness']
            
            if abs(skew_val) > 1:
                # Asymétrie forte -> transformation forte
                if (df[col] > 0).all():
                    df_transformed = transformer.box_cox_transform(df_transformed, [col])
                else:
                    df_transformed = transformer.yeo_johnson_transform(df_transformed, [col])
            elif abs(skew_val) > 0.5:
                # Asymétrie modérée -> transformation modérée
                if (df[col] >= 0).all():
                    df_transformed = transformer.sqrt_transform(df_transformed, [col])
                else:
                    df_transformed = transformer.yeo_johnson_transform(df_transformed, [col])
    
    elif method == 'log':
        df_transformed = transformer.log_transform(df, columns)
    elif method == 'sqrt':
        df_transformed = transformer.sqrt_transform(df, columns)
    elif method == 'boxcox':
        df_transformed = transformer.box_cox_transform(df, columns)
    elif method == 'yeo-johnson':
        df_transformed = transformer.yeo_johnson_transform(df, columns)
    elif method == 'quantile':
        df_transformed = transformer.quantile_transform(df, columns)
    else:
        df_transformed = df.copy()
    
    return df_transformed, transformer
