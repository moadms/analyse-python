# 🏥 Analyse Prédictive des AVC (Stroke) - Application ML

Application d'analyse et de prédiction des AVC utilisant le Machine Learning, développée avec Streamlit et Python.

## 📋 Description

Ce projet est une application web interactive pour l'analyse prédictive des accidents vasculaires cérébraux (AVC). Il permet d'explorer les données, d'appliquer différentes techniques de prétraitement et d'équilibrage, et d'entraîner plusieurs modèles de Machine Learning pour prédire le risque d'AVC.

## 🎯 Fonctionnalités Principales

### 📊 Analyse des Données
- Chargement et exploration du dataset (5,110 patients)
- Détection et gestion des valeurs manquantes
- Analyse statistique descriptive
- Visualisations interactives

### 🔧 Prétraitement
- Encodage des variables catégorielles
- Normalisation/Standardisation
- Détection et traitement des outliers
- Transformations de variables (Log, Box-Cox, Yeo-Johnson)

### ⚖️ Équilibrage des Classes
Comparaison de **10 techniques** d'équilibrage :
- Random Oversampling / Undersampling
- SMOTE (Synthetic Minority Oversampling)
- ADASYN (Adaptive Synthetic Sampling)
- Tomek Links
- NearMiss
- SMOTE + Tomek
- SMOTE + ENN
- Borderline SMOTE
- SVM SMOTE

### 📈 Analyse Exploratoire Avancée
- **PCA** (Analyse en Composantes Principales)
- **ACM** (Analyse des Correspondances Multiples)
- **AFD** (Analyse Factorielle Discriminante)

### 🤖 Modèles Machine Learning
Entraînement et comparaison de **6 algorithmes** :
1. **XGBoost** (Extreme Gradient Boosting)
2. Random Forest
3. Gradient Boosting
4. Logistic Regression
5. SVM (Support Vector Machine)
6. AdaBoost

### ✅ Validation Croisée
- **5-Fold StratifiedKFold** pour validation robuste
- Métriques de performance complètes

### 📄 Génération de Rapports
- Rapports Markdown/HTML personnalisables
- Export Excel avec métriques ML
- Export CSV des données
- Organisation professionnelle des résultats

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/moadms/analyse-python.git
cd analyse-python
```

2. **Créer un environnement virtuel**
```bash
python -m venv .venv
```

3. **Activer l'environnement virtuel**

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Windows (CMD):
```cmd
.venv\Scripts\activate.bat
```

Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Installer les dépendances**
```bash
cd stroke-ml-app
pip install -r requirements.txt
```

## 📦 Lancement de l'Application

### Méthode 1 : Script batch (Windows)
```bash
cd stroke-ml-app
start.bat
```

### Méthode 2 : Ligne de commande
```bash
cd stroke-ml-app
streamlit run app.py
```

L'application sera accessible sur : **http://localhost:8501**

## 📁 Structure du Projet

```
analyse-python/
│
├── stroke-ml-app/           # Dossier principal de l'application
│   ├── app.py               # Application Streamlit principale (5,700+ lignes)
│   ├── requirements.txt     # Dépendances Python
│   ├── start.bat            # Script de lancement Windows
│   │
│   ├── utils/               # Modules utilitaires
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py    # Prétraitement des données
│   │   ├── transformations.py       # Transformations de variables
│   │   ├── balancing.py             # Techniques d'équilibrage
│   │   ├── models.py                # Entraînement des modèles ML
│   │   └── visualizations.py        # Fonctions de visualisation
│   │
│   ├── .streamlit/          # Configuration Streamlit
│   │   └── config.toml
│   │
│   ├── healthcare-dataset-stroke-data.csv  # Dataset
│   │
│   └── Documentation/
│       ├── README.md                      # Guide du projet
│       ├── GUIDE_UTILISATION.md           # Guide utilisateur
│       ├── AMELIORATIONS_FINALES.md       # Documentation technique
│       └── PRESENTATION.md                # Support de présentation
│
└── README.md                # Ce fichier
```

## 📊 Dataset

- **Source** : Healthcare Dataset - Stroke Data
- **Taille** : 5,110 patients
- **Variables** : 12 caractéristiques
- **Cible** : Stroke (0 = Pas d'AVC, 1 = AVC)
- **Déséquilibre** : 4.87% de cas positifs (1:19.5)

### Variables Principales
- Âge, Genre, Hypertension, Maladies cardiaques
- État marital, Type de travail, Type de résidence
- Glucose moyen, BMI (Indice de Masse Corporelle)
- Statut fumeur

## 🎓 Contexte Académique

Projet développé dans le cadre du cursus de **4ème année Génie Informatique**.

**Auteur** : MOAD MOUSSAOUI  
**Année** : 2024-2025

## 📈 Métriques de Performance

L'application calcule automatiquement :
- **Accuracy** (Précision globale)
- **Precision** (Précision positive)
- **Recall** (Sensibilité)
- **F1-Score** (Moyenne harmonique)
- **AUC-ROC** (Aire sous la courbe)
- **Matrice de confusion**
- **Temps d'entraînement**

## 🛠️ Technologies Utilisées

### Librairies Principales
- **Streamlit** : Interface web interactive
- **Pandas & NumPy** : Manipulation de données
- **Scikit-learn** : Modèles ML et prétraitement
- **XGBoost** : Gradient Boosting avancé
- **Imbalanced-learn** : Techniques d'équilibrage
- **Plotly** : Visualisations interactives
- **Seaborn & Matplotlib** : Graphiques statiques
- **Prince** : ACM (Analyse des Correspondances Multiples)
- **openpyxl** : Export Excel

## 🔍 Workflows Recommandés

### Workflow Minimal (20 min)
1. Exploration des Données (Étape 1)
2. Prétraitement Simple (Étape 4)
3. Équilibrage SMOTE (Étape 6)
4. Entraînement XGBoost (Étape 8)

### Workflow Complet (60 min)
1. Exploration Approfondie (Étape 1)
2. Analyse des Outliers (Étape 2)
3. Transformations (Étape 3)
4. Prétraitement Avancé (Étape 4)
5. Visualisations (Étape 5)
6. Comparaison des 10 Techniques (Étape 6)
7. ACM + PCA + AFD (Étapes 7a-c)
8. Tous les Modèles + CV (Étape 8)
9. Génération de Rapports (Étape 13)

## 📝 Améliorations Récentes

✅ Ajout de XGBoost et AdaBoost  
✅ Validation Croisée 5-Fold  
✅ Dashboard de comparaison des techniques d'équilibrage  
✅ Analyse Factorielle Discriminante (AFD)  
✅ Génération de rapports professionnels  
✅ Correction des bugs (ACM, AFD, BMI display)  
✅ Architecture modulaire optimisée  

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est développé à des fins académiques.

## 📧 Contact

**MOAD MOUSSAOUI**  
GitHub: [@moadms](https://github.com/moadms)

---

⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !
