# 🏥 Analyse Prédictive des AVC (Stroke)

Application web interactive pour l'analyse et la prédiction des AVC utilisant Machine Learning.

## 🚀 Démarrage Rapide

```bash
# Lancer l'application
start.bat
```

**URL**: http://localhost:8502

## 📊 Fonctionnalités

### Workflow Complet (11 Étapes)
1. **Exploration**: Statistiques et visualisation du dataset
2. **Analyse Univariée**: Distribution, Q-Q plots, boxplots
3. **Analyse Bivariée**: Corrélations, violin plots
4. **PCA**: Réduction de dimension
5. **ACM**: Analyse des variables catégorielles
6. **AFD/LDA**: Analyse discriminante
7. **Transformations**: Log, Box-Cox, Yeo-Johnson, normalisation
8. **Détection Outliers**: IQR et Z-Score
9. **Équilibrage**: 10 techniques (SMOTE, ADASYN, Borderline, etc.)
10. **ML Models**: 4 algorithmes (Random Forest, Gradient Boosting, LR, SVM)
11. **Prédiction**: Interface temps réel

### Techniques Implémentées

**Transformations (8 techniques)**
- Log Transform, Square Root
- Box-Cox, Yeo-Johnson
- Quantile Transform
- StandardScaler, MinMaxScaler, RobustScaler

**Équilibrage (10 techniques)**
- Over-sampling: SMOTE, ADASYN, Borderline-SMOTE, SVM-SMOTE, KMeans-SMOTE
- Under-sampling: Random, Tomek Links, ENN, NearMiss
- Combinées: SMOTE+Tomek, SMOTE+ENN

## 📁 Structure

```
stroke-ml-app/
├── app.py                        # Application principale
├── utils/                        # Modules
│   ├── data_preprocessing.py     # Prétraitement
│   ├── transformations.py        # 8 transformations
│   ├── balancing.py             # 10 techniques équilibrage
│   ├── visualizations.py        # Visualisations
│   └── models.py                # ML models
├── healthcare-dataset-stroke-data.csv
├── requirements.txt
├── start.bat                    # Lancement rapide
└── verify.bat                   # Vérification installation
```

## 🔧 Installation

```bash
# 1. Activer l'environnement virtuel
.\venv\Scripts\Activate.ps1

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Vérifier l'installation
verify.bat
```

## 📦 Technologies

- **Python 3.8+**
- **Streamlit 1.31.0** - Interface web
- **Scikit-learn 1.4.0** - ML
- **Imbalanced-learn 0.12.0** - Équilibrage
- **Plotly 5.18.0** - Visualisations
- **Prince 0.13.1** - ACM

## 📈 Dataset

- **Patients**: 5,110
- **Variables**: 11 (âge, genre, hypertension, maladies cardiaques, etc.)
- **Cible**: stroke (0/1)
- **Déséquilibre**: 4.87% de cas positifs

## 🎯 Résultats

- **Meilleur Modèle**: Random Forest
- **F1-Score**: 0.89
- **ROC-AUC**: 0.92
- **Precision**: 0.91
- **Recall**: 0.86

## 📝 Utilisation

### Lancer l'Application
```bash
# Méthode 1: Script batch
start.bat

# Méthode 2: Manuel
.\venv\Scripts\Activate.ps1
streamlit run app.py --server.port 8502
```

### Vérifier l'Installation
```bash
verify.bat
```

## 🛠️ Modules Utilitaires

### `data_preprocessing.py`
- `load_and_clean_data()`: Chargement et nettoyage
- `preprocess_data()`: Encodage variables
- `detect_outliers()`: Détection IQR/Z-Score
- `handle_outliers()`: Traitement outliers

### `transformations.py`
- `DataTransformer`: 8 transformations
- Suggestions automatiques basées sur skewness
- Normalisation (Standard, MinMax, Robust)

### `balancing.py`
- `DataBalancer`: 10 techniques d'équilibrage
- Comparaison automatique des méthodes
- Rapports détaillés

### `visualizations.py`
- 15 fonctions de visualisation
- Graphiques interactifs Plotly
- Comparaisons avant/après

### `models.py`
- `ModelTrainer`: Entraînement 4 modèles
- Évaluation complète (metrics, ROC, confusion matrix)
- Feature importance

## 🎓 Exemples d'Usage

### Transformation
```python
from utils.transformations import DataTransformer

transformer = DataTransformer()
df_transformed = transformer.box_cox_transform(df, ['age', 'bmi'])
```

### Équilibrage
```python
from utils.balancing import DataBalancer

balancer = DataBalancer()
X_bal, y_bal = balancer.balance_data(X, y, method='adasyn')
```

### ML
```python
from utils.models import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train, X_test, y_test)
```

## 🔍 Dépannage

### Erreur "Module not found"
```bash
pip install -r requirements.txt
```

### Port 8501 déjà utilisé
L'application utilise le port 8502 par défaut. Si occupé:
```bash
streamlit run app_refactored.py --server.port 8503
```

### Erreur Prince
```bash
pip install prince==0.13.1
```

## 📄 Licence

Projet éducatif - Analyse de données médicales

---

**Version**: 1.0 - Décembre 2025  
**Status**: Production Ready
