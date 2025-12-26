"""
Module de rééquilibrage des classes (SMOTE, ADASYN, etc.).
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours, 
    NearMiss, CondensedNearestNeighbour
)
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter


class DataBalancer:
    """Classe pour gérer le rééquilibrage des données."""
    
    def __init__(self):
        self.balancer = None
        self.method_used = None
        self.original_distribution = None
        self.balanced_distribution = None
    
    def get_class_distribution(self, y):
        """Retourne la distribution des classes."""
        return dict(Counter(y))
    
    def smote_oversampling(self, X, y, sampling_strategy='auto', k_neighbors=5):
        """
        SMOTE (Synthetic Minority Over-sampling Technique)
        Génère des exemples synthétiques de la classe minoritaire
        """
        self.balancer = SMOTE(sampling_strategy=sampling_strategy, 
                             k_neighbors=k_neighbors, 
                             random_state=42)
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        self.method_used = 'SMOTE'
        
        return X_balanced, y_balanced
    
    def adasyn_oversampling(self, X, y, sampling_strategy='auto', n_neighbors=5):
        """
        ADASYN (Adaptive Synthetic Sampling)
        Version adaptative de SMOTE - génère plus d'exemples pour les zones difficiles
        """
        try:
            self.balancer = ADASYN(sampling_strategy=sampling_strategy, 
                                  n_neighbors=n_neighbors, 
                                  random_state=42)
            X_balanced, y_balanced = self.balancer.fit_resample(X, y)
            self.method_used = 'ADASYN'
            return X_balanced, y_balanced
        except Exception as e:
            print(f"ADASYN failed: {e}. Falling back to SMOTE.")
            return self.smote_oversampling(X, y, sampling_strategy, n_neighbors)
    
    def borderline_smote(self, X, y, sampling_strategy='auto', k_neighbors=5):
        """
        Borderline-SMOTE
        Se concentre sur les exemples à la frontière entre classes
        """
        self.balancer = BorderlineSMOTE(sampling_strategy=sampling_strategy, 
                                       k_neighbors=k_neighbors, 
                                       random_state=42)
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        self.method_used = 'Borderline-SMOTE'
        
        return X_balanced, y_balanced
    
    def svm_smote(self, X, y, sampling_strategy='auto', k_neighbors=5):
        """
        SVM-SMOTE
        Utilise un SVM pour identifier les exemples de frontière
        """
        try:
            self.balancer = SVMSMOTE(sampling_strategy=sampling_strategy, 
                                    k_neighbors=k_neighbors, 
                                    random_state=42)
            X_balanced, y_balanced = self.balancer.fit_resample(X, y)
            self.method_used = 'SVM-SMOTE'
            return X_balanced, y_balanced
        except:
            return self.smote_oversampling(X, y, sampling_strategy, k_neighbors)
    
    def kmeans_smote(self, X, y, sampling_strategy='auto', k_neighbors=5):
        """
        KMeans-SMOTE
        Utilise KMeans pour créer des clusters avant SMOTE
        """
        try:
            self.balancer = KMeansSMOTE(sampling_strategy=sampling_strategy, 
                                       k_neighbors=k_neighbors, 
                                       random_state=42)
            X_balanced, y_balanced = self.balancer.fit_resample(X, y)
            self.method_used = 'KMeans-SMOTE'
            return X_balanced, y_balanced
        except:
            return self.smote_oversampling(X, y, sampling_strategy, k_neighbors)
    
    def random_undersampling(self, X, y, sampling_strategy='auto'):
        """
        Random Under-sampling
        Supprime aléatoirement des exemples de la classe majoritaire
        """
        self.balancer = RandomUnderSampler(sampling_strategy=sampling_strategy, 
                                          random_state=42)
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        self.method_used = 'Random Under-sampling'
        
        return X_balanced, y_balanced
    
    def tomek_links(self, X, y):
        """
        Tomek Links
        Supprime les paires d'exemples trop proches de classes différentes
        """
        self.balancer = TomekLinks()
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        self.method_used = 'Tomek Links'
        
        return X_balanced, y_balanced
    
    def edited_nearest_neighbours(self, X, y, n_neighbors=3):
        """
        Edited Nearest Neighbours (ENN)
        Supprime les exemples mal classés par leurs k voisins
        """
        self.balancer = EditedNearestNeighbours(n_neighbors=n_neighbors)
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        self.method_used = 'ENN'
        
        return X_balanced, y_balanced
    
    def near_miss(self, X, y, version=1):
        """
        NearMiss
        Sélectionne les exemples de la classe majoritaire les plus proches de la minoritaire
        """
        try:
            self.balancer = NearMiss(version=version)
            X_balanced, y_balanced = self.balancer.fit_resample(X, y)
            self.method_used = f'NearMiss-{version}'
            return X_balanced, y_balanced
        except:
            return self.random_undersampling(X, y)
    
    def smote_tomek(self, X, y, sampling_strategy='auto'):
        """
        SMOTE + Tomek Links (Combinaison)
        Over-sampling avec SMOTE puis nettoyage avec Tomek
        """
        self.balancer = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        self.method_used = 'SMOTE+Tomek'
        
        return X_balanced, y_balanced
    
    def smote_enn(self, X, y, sampling_strategy='auto'):
        """
        SMOTE + ENN (Combinaison)
        Over-sampling avec SMOTE puis nettoyage avec ENN
        """
        self.balancer = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        self.method_used = 'SMOTE+ENN'
        
        return X_balanced, y_balanced
    
    def balance_data(self, X, y, method='smote', **kwargs):
        """
        Méthode générique pour équilibrer les données.
        
        Parameters:
        -----------
        method : str
            'smote', 'adasyn', 'borderline', 'svm-smote', 'kmeans-smote',
            'random-under', 'tomek', 'enn', 'nearmiss', 'smote-tomek', 'smote-enn'
        """
        self.original_distribution = self.get_class_distribution(y)
        
        methods = {
            'smote': self.smote_oversampling,
            'adasyn': self.adasyn_oversampling,
            'borderline': self.borderline_smote,
            'svm-smote': self.svm_smote,
            'kmeans-smote': self.kmeans_smote,
            'random-under': self.random_undersampling,
            'tomek': self.tomek_links,
            'enn': self.edited_nearest_neighbours,
            'nearmiss': self.near_miss,
            'smote-tomek': self.smote_tomek,
            'smote-enn': self.smote_enn
        }
        
        if method not in methods:
            raise ValueError(f"Méthode inconnue: {method}")
        
        X_balanced, y_balanced = methods[method](X, y, **kwargs)
        self.balanced_distribution = self.get_class_distribution(y_balanced)
        
        return X_balanced, y_balanced
    
    def get_balancing_report(self):
        """Retourne un rapport sur le rééquilibrage."""
        report = {
            'method': self.method_used,
            'original': self.original_distribution,
            'balanced': self.balanced_distribution,
            'original_ratio': self.original_distribution.get(1, 0) / self.original_distribution.get(0, 1),
            'balanced_ratio': self.balanced_distribution.get(1, 0) / self.balanced_distribution.get(0, 1)
        }
        return report


def compare_balancing_methods(X, y, methods=['smote', 'adasyn', 'borderline', 'smote-tomek']):
    """
    Compare plusieurs méthodes de rééquilibrage.
    
    Returns:
    --------
    dict : {method: (X_balanced, y_balanced, report)}
    """
    results = {}
    
    for method in methods:
        balancer = DataBalancer()
        try:
            X_bal, y_bal = balancer.balance_data(X, y, method=method)
            report = balancer.get_balancing_report()
            results[method] = {
                'X': X_bal,
                'y': y_bal,
                'report': report
            }
        except Exception as e:
            print(f"Erreur avec {method}: {e}")
            results[method] = None
    
    return results
