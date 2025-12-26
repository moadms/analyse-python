# 🏥 Application d'Analyse et Prédiction des AVC (Stroke)

## 📋 Description

Application web moderne et interactive développée avec **Streamlit** pour l'analyse complète du dataset Healthcare Stroke et la prédiction des risques d'AVC à l'aide du Machine Learning.

Cette application implémente toutes les étapes d'une analyse de données scientifique complète, de la recherche bibliographique jusqu'à la mise en production de modèles prédictifs.

## ✨ Fonctionnalités

### 📖 Étape 1: Recherche Scientifique
- Présentation du dataset et des variables
- Tableau des algorithmes de prétraitement et ML proposés
- Métriques d'évaluation
- Aperçu des données

### 📊 Étape 2: Analyse Univariée
- **Variables Numériques:**
  - Statistiques descriptives complètes (moyenne, médiane, variance, skewness, kurtosis)
  - Distribution des fréquences (histogrammes interactifs)
  - QQ-Plots pour tester la normalité
  - Boxplots pour identifier les valeurs aberrantes

- **Variables Catégorielles:**
  - Graphiques en camembert
  - Diagrammes en barres

- **Gestion des Valeurs Manquantes:**
  - Identification et quantification
  - Stratégies d'imputation (médiane pour BMI)

- **Détection des Valeurs Aberrantes:**
  - Méthode IQR (Interquartile Range)
  - Visualisations boxplot
  - Justifications pour conservation/suppression

### 🔗 Étape 3: Analyse Bivariée
- **Matrices de Corrélation:**
  - Heatmap complète avec toutes les variables
  - Corrélations avec la variable cible (stroke)
  - Identification des variables les plus prédictives

- **Visualisations:**
  - Violin plots (âge vs stroke, glucose vs stroke)
  - Graphiques croisés pour variables catégorielles
  - Analyse des relations entre variables

- **Décisions et Justifications:**
  - Documentation des transformations appliquées
  - Variables conservées/supprimées avec justifications

### 🎯 Étape 4: Analyse Multivariée (PCA/ACM)
- **Analyse en Composantes Principales (PCA):**
  - Calcul de la variance expliquée par composante
  - Variance cumulée avec seuil 80%
  - Projection des individus sur PC1 et PC2
  - Cercle de corrélation (projection des variables)
  - Visualisation 3D (PC1, PC2, PC3)

- **Interprétation:**
  - Identification des axes principaux
  - Groupes d'individus
  - Tendances des modalités

### ⚖️ Gestion du Déséquilibre des Données
- **Analyse du déséquilibre initial:**
  - Ratio de déséquilibre
  - Visualisations avant/après

- **Techniques de rééquilibrage:**
  - **SMOTE** (Synthetic Minority Over-sampling Technique)
  - **Random Under-Sampling**
  - **SMOTE + Tomek Links**

- **Comparaisons visuelles:**
  - Graphiques comparatifs
  - Statistiques détaillées

### 🤖 Modèles ML & Prédiction
- **Algorithmes implémentés:**
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)

- **Métriques d'évaluation:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
  - Cross-Validation

- **Visualisations:**
  - Matrices de confusion pour chaque modèle
  - Courbes ROC comparatives
  - Feature Importance (Random Forest)
  - Comparaison des performances

- **Interface de Prédiction:**
  - Formulaire interactif pour entrer les données patient
  - Prédiction en temps réel
  - Jauge de risque visuelle
  - Recommandations personnalisées

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes d'installation

1. **Cloner ou télécharger le projet:**
```bash
cd "D:\Stroke analyse\stroke-ml-app"
```

2. **Créer un environnement virtuel (recommandé):**
```bash
python -m venv venv
```

3. **Activer l'environnement virtuel:**
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. **Installer les dépendances:**
```bash
pip install -r requirements.txt
```

5. **Copier le dataset:**
Assurez-vous que le fichier `healthcare-dataset-stroke-data.csv` est présent dans le même répertoire que `app.py`.

## 📊 Utilisation

1. **Lancer l'application:**
```bash
streamlit run app.py
```

2. **Accéder à l'application:**
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse: `http://localhost:8501`

3. **Navigation:**
- Utilisez le menu latéral pour naviguer entre les différentes étapes
- Explorez les visualisations interactives
- Testez les modèles ML
- Faites des prédictions personnalisées

## 📁 Structure du Projet

```
stroke-ml-app/
│
├── app.py                              # Application principale Streamlit
├── requirements.txt                     # Dépendances Python
├── README.md                           # Ce fichier
├── healthcare-dataset-stroke-data.csv  # Dataset (à copier)
└── venv/                               # Environnement virtuel (créé lors de l'installation)
```

## 🔬 Dataset

**Source:** Healthcare Dataset Stroke Data

**Variables:**
- `id`: Identifiant unique
- `gender`: Genre (Male/Female/Other)
- `age`: Âge du patient
- `hypertension`: Hypertension (0=Non, 1=Oui)
- `heart_disease`: Maladie cardiaque (0=Non, 1=Oui)
- `ever_married`: Statut marital (Yes/No)
- `work_type`: Type de travail (children, Govt_job, Never_worked, Private, Self-employed)
- `Residence_type`: Type de résidence (Rural/Urban)
- `avg_glucose_level`: Niveau moyen de glucose
- `bmi`: Indice de masse corporelle
- `smoking_status`: Statut de fumeur (formerly smoked, never smoked, smokes, Unknown)
- `stroke`: Variable cible (0=Non, 1=Oui)

**Statistiques:**
- **Lignes:** 5,110
- **Colonnes:** 12
- **Cas de Stroke:** 249 (4.87%)
- **Déséquilibre:** Ratio 1:19.5

## 🎨 Design

L'application utilise un design moderne avec:
- Gradient violette (#667eea à #764ba2)
- Cartes métriques stylisées
- Graphiques interactifs Plotly
- Interface responsive
- Thème cohérent et professionnel

## 📈 Résultats Attendus

L'application permet de:
1. ✅ Comprendre complètement le dataset
2. ✅ Identifier les facteurs de risque principaux
3. ✅ Visualiser les relations entre variables
4. ✅ Équilibrer les données déséquilibrées
5. ✅ Comparer plusieurs algorithmes ML
6. ✅ Prédire le risque d'AVC pour de nouveaux patients
7. ✅ Obtenir des recommandations médicales

## 🛠️ Technologies Utilisées

- **Streamlit**: Framework web pour applications de data science
- **Pandas**: Manipulation de données
- **NumPy**: Calculs numériques
- **Plotly**: Visualisations interactives
- **Seaborn/Matplotlib**: Visualisations statistiques
- **Scikit-learn**: Algorithmes de Machine Learning
- **Imbalanced-learn**: Gestion du déséquilibre des données
- **SciPy**: Calculs scientifiques

## 📝 Notes pour la Présentation

### Points forts à mentionner:
1. **Interface moderne et intuitive** - Design professionnel avec gradients
2. **Analyse complète** - Toutes les étapes du workflow ML
3. **Interactivité** - Graphiques Plotly dynamiques
4. **Gestion du déséquilibre** - Crucial pour ce dataset (ratio 1:19.5)
5. **Comparaison de modèles** - 4 algorithmes différents
6. **Prédiction en temps réel** - Interface utilisateur pratique

### Démonstration suggérée:
1. Commencer par Étape 1 (présenter le dataset)
2. Montrer l'analyse univariée (statistiques + visualisations)
3. Expliquer le déséquilibre et montrer les techniques SMOTE
4. Entraîner les modèles et montrer les performances
5. Faire une prédiction live avec un cas exemple
6. Montrer les courbes ROC et matrices de confusion

## 🔧 Dépannage

### Problème: Module non trouvé
```bash
pip install --upgrade -r requirements.txt
```

### Problème: Port déjà utilisé
```bash
streamlit run app.py --server.port 8502
```

### Problème: Dataset non trouvé
Assurez-vous que `healthcare-dataset-stroke-data.csv` est dans le même dossier que `app.py`.

## 📧 Support

Pour toute question ou problème, consultez la documentation Streamlit: https://docs.streamlit.io

## 🎓 Objectifs Pédagogiques

Cette application démontre:
- ✅ Maîtrise de l'analyse de données
- ✅ Compréhension du Machine Learning
- ✅ Capacité à gérer des données déséquilibrées
- ✅ Compétences en visualisation de données
- ✅ Développement d'applications web interactives
- ✅ Présentation de résultats scientifiques

## 📜 Licence

Projet éducatif - Libre d'utilisation pour apprentissage et présentation.

---

**Développé avec ❤️ pour l'analyse médicale et le Machine Learning**

**Bonne chance pour votre présentation! 🎉**
#   a n a l y s e - d o n n - e s  
 