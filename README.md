# 🏥 Projet d'Analyse Prédictive des AVC (Stroke)

Créé par **MOAD MOUSSAOUI** | Projet 4ème Année Génie Informatique

## 🚀 Démarrage Rapide

### Méthode 1: Script de Démarrage (Recommandé)
```bash
# Double-cliquez sur le fichier
start_app.bat
```

### Méthode 2: Manuel
```bash
# Naviguer vers le dossier du projet
cd stroke-ml-app

# Activer l'environnement virtuel
..\.venv\Scripts\activate

# Lancer l'application
streamlit run app.py
```

## 📂 Structure du Projet

```
d:\Stroke analyse\
├── .venv/                     # Environnement Python virtuel
├── .git/                      # Git repository
├── stroke-ml-app/             # 📁 APPLICATION PRINCIPALE
│   ├── app.py                 # 🚀 Fichier principal Streamlit
│   ├── healthcare-dataset-stroke-data.csv  # 📊 Dataset (5,110 patients)
│   ├── utils/                 # 🔧 Modules utilitaires
│   │   ├── data_preprocessing.py
│   │   ├── balancing.py
│   │   ├── models.py
│   │   ├── transformations.py
│   │   └── visualizations.py
│   ├── requirements.txt       # 📦 Dépendances Python
│   ├── start.bat             # 🚀 Script de démarrage
│   └── README.md             # 📖 Documentation détaillée
└── start_app.bat             # 🚀 Script global de démarrage
```

## ✨ Fonctionnalités

### 🎯 13 Étapes de Workflow Data Science
- **Recherche Scientifique**: Revue de littérature (5 articles)
- **Exploration des Données**: Aperçu, statistiques, distribution cible
- **Analyse Univariée**: Distribution de toutes les variables
- **Analyse Bivariée**: Corrélations et relations
- **Prétraitement**: Imputation, gestion outliers
- **Transformations**: 8 techniques de normalisation
- **Équilibrage**: 10 méthodes (SMOTE, ADASYN, etc.)
- **ACP**: Analyse Composantes Principales
- **ACM**: Analyse Correspondances Multiples
- **AFD**: Analyse Factorielle Discriminante
- **Modèles ML**: 6 algorithmes de classification
- **Prédiction**: Interface temps réel
- **Rapports**: Génération automatique

### 🤖 Modèles ML Disponibles
1. **Logistic Regression** (baseline)
2. **Random Forest** 
3. **Gradient Boosting**
4. **XGBoost** ⭐ (meilleur performer)
5. **Support Vector Machine**
6. **AdaBoost**

### ⚖️ Techniques d'Équilibrage
- **SMOTE** (baseline)
- **ADASYN** ⭐ (adaptatif) 
- **BorderlineSMOTE**
- **SVMSMOTE**
- **SMOTETomek** (hybride)
- **SMOTEENN** (hybride)
- **TomekLinks** (sous-échantillonnage)
- **EditedNearestNeighbours**
- **RandomUnderSampler**
- **RandomOverSampler**

## 📊 Dataset

- **Source**: Healthcare Dataset Stroke Data (Kaggle)
- **Taille**: 5,110 patients
- **Variables**: 11 (10 prédicteurs + 1 cible)
- **Déséquilibre**: 4.87% stroke (1:19.5 ratio)
- **Variables clés**: age, avg_glucose_level, bmi, hypertension

## 🎯 Performances Attendues

Basé sur benchmarks scientifiques (même dataset):
- **Accuracy**: ≥ 95%
- **Recall**: ≥ 85% (crucial pour médical)
- **Precision**: ≥ 90%
- **F1-Score**: ≥ 92%
- **ROC-AUC**: ≥ 0.95

## 💻 Technologies Utilisées

### 📦 Python Packages
- **Streamlit**: Interface web interactive
- **XGBoost**: Gradient boosting optimisé
- **scikit-learn**: ML algorithms
- **imbalanced-learn**: Techniques d'équilibrage
- **pandas/numpy**: Manipulation données
- **plotly/seaborn**: Visualisations
- **prince**: Analyses factorielles

### 🏗️ Infrastructure
- **Python 3.14.0**: Latest version
- **Virtual Environment**: .venv isolé
- **Git**: Version control
- **Streamlit**: Web framework

## 🚀 URL d'Accès

Une fois lancé, l'application sera disponible sur:
**http://localhost:8501**

## 🏥 Application Médicale

Cette application est conçue pour aider les professionnels de santé à:
- **Prédire le risque d'AVC** chez les patients
- **Identifier les facteurs de risque** principaux
- **Évaluer l'impact des interventions** préventives
- **Générer des rapports** pour comités éthiques

## 🎓 Contexte Académique

Projet réalisé dans le cadre du cursus **4ème Année Génie Informatique**.

**Objectifs pédagogiques**:
- Maîtrise complete du pipeline ML
- Application des techniques d'équilibrage
- Analyses multivariées avancées
- Interface utilisateur professionnelle
- Documentation scientifique

---

**🔬 Basé sur la recherche scientifique** | **🚀 Production-ready** | **🏥 Usage médical**
