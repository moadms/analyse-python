"""
Module d'entraînement et d'évaluation des modèles ML.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)


class ModelTrainer:
    """Classe pour entraîner et évaluer des modèles ML."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Divise les données en train/test."""
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    
    def train_random_forest(self, X_train, y_train, **params):
        """Entraîne un Random Forest."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        default_params.update(params)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, **params):
        """Entraîne un Gradient Boosting."""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        default_params.update(params)
        
        model = GradientBoostingClassifier(**default_params)
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        
        return model
    
    def train_logistic_regression(self, X_train, y_train, **params):
        """Entraîne une Régression Logistique."""
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        default_params.update(params)
        
        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        
        return model
    
    def train_svm(self, X_train, y_train, **params):
        """Entraîne un SVM."""
        default_params = {
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        default_params.update(params)
        
        model = SVC(**default_params)
        model.fit(X_train, y_train)
        self.models['SVM'] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """Évalue un modèle sur les données de test."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = metrics
        
        # Mettre à jour le meilleur modèle
        if metrics['f1_score'] > self.best_score:
            self.best_score = metrics['f1_score']
            self.best_model = model
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Entraîne tous les modèles et les évalue."""
        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        self.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        
        # Gradient Boosting
        gb_model = self.train_gradient_boosting(X_train, y_train)
        self.evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting')
        
        # Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        self.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
        
        # SVM
        svm_model = self.train_svm(X_train, y_train)
        self.evaluate_model(svm_model, X_test, y_test, 'SVM')
        
        return self.results
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Retourne l'importance des features."""
        model = self.models.get(model_name)
        
        if model and hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif model and hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None
    
    def cross_validate(self, model, X, y, cv=5, scoring='f1'):
        """Effectue une validation croisée."""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores.tolist()
        }
    
    def hyperparameter_tuning(self, model_type, X_train, y_train, param_grid, cv=5):
        """Optimise les hyperparamètres avec GridSearchCV."""
        models_mapping = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True)
        }
        
        base_model = models_mapping.get(model_type)
        if not base_model:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
        
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, 
                                   scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def get_results_dataframe(self):
        """Retourne les résultats sous forme de DataFrame."""
        df_results = pd.DataFrame.from_dict(self.results, orient='index')
        df_results = df_results[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        return df_results.round(4)
    
    def predict(self, model_name, X):
        """Fait une prédiction avec un modèle spécifique."""
        model = self.models.get(model_name)
        if model:
            return model.predict(X), model.predict_proba(X)
        return None, None
