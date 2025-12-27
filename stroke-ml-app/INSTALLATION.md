# 📋 Guide d'Installation - Application Prédiction AVC

## 👨‍💻 Créateur: MOAD MOUSSAOUI
## 📅 Date: 27 Décembre 2025

---

## 🚀 Installation Rapide

### 1️⃣ Prérequis
- **Python 3.8+** installé sur votre système
- **Connexion Internet** (pour téléchargement dataset)

### 2️⃣ Installation des Dépendances

```bash
# Naviguer vers le dossier du projet
cd "d:\Stroke analyse\stroke-ml-app"

# Installer les packages Python requis
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib imbalanced-learn prince scipy xgboost
```

### 3️⃣ Dataset

**Option A: Téléchargement Automatique (Recommandé)**
- L'application téléchargera automatiquement le dataset au premier lancement
- Source: GitHub mirror du Kaggle Healthcare Stroke Dataset

**Option B: Téléchargement Manuel**

Si le téléchargement automatique échoue:

1. **Téléchargez le fichier depuis Kaggle:**
   - URL: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
   - Ou: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/healthcare-dataset-stroke-data.csv

2. **Placez le fichier dans le bon dossier:**
   ```
   d:\Stroke analyse\stroke-ml-app\healthcare-dataset-stroke-data.csv
   ```

3. **Vérifiez le chemin:**
   - Le fichier doit être à côté de `app.py`
   - Nom exact: `healthcare-dataset-stroke-data.csv`

### 4️⃣ Lancement de l'Application

```bash
# Méthode 1: Avec Python
python -m streamlit run app.py

# Méthode 2: Avec le fichier batch (Windows)
start.bat

# Méthode 3: Avec environnement virtuel
.\.venv\Scripts\python.exe -m streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse:
```
http://localhost:8501
```

---

## 🗂️ Structure du Projet

```
stroke-ml-app/
│
├── app.py                                  # Application principale Streamlit
├── healthcare-dataset-stroke-data.csv      # Dataset (téléchargé automatiquement)
│
├── utils/                                  # Modules utilitaires
│   ├── __init__.py
│   ├── balancing.py                        # SMOTE, ADASYN, etc.
│   ├── data_preprocessing.py               # Nettoyage, imputation
│   ├── models.py                           # Modèles ML
│   ├── transformations.py                  # Normalisation, encodage
│   └── visualizations.py                   # Graphiques Plotly
│
├── requirements.txt                        # Dépendances Python
├── start.bat                               # Lanceur Windows
├── README.md                               # Documentation principale
├── INSTALLATION.md                         # Ce fichier
└── RESUME_AMELIORATIONS_FINALES.md        # Documentation complète

```

---

## 📦 Dépendances Complètes

### Core Framework:
- **streamlit** (v1.28+) - Interface web interactive
- **pandas** (v2.1+) - Manipulation données
- **numpy** (v1.24+) - Calculs numériques

### Machine Learning:
- **scikit-learn** (v1.3+) - Modèles ML de base
- **xgboost** (v2.0+) - Gradient Boosting optimisé
- **imbalanced-learn** (v0.11+) - SMOTE, ADASYN

### Visualisation:
- **plotly** (v5.17+) - Graphiques interactifs
- **seaborn** (v0.13+) - Heatmaps
- **matplotlib** (v3.8+) - Graphiques statiques

### Analyses Multivariées:
- **prince** (v0.7+) - ACM
- **scipy** (v1.11+) - AFD, statistiques

---

## 🐛 Résolution des Problèmes

### Erreur: "FileNotFoundError: healthcare-dataset-stroke-data.csv"

**Solution 1: Téléchargement automatique**
- Relancez l'application - elle téléchargera automatiquement le fichier

**Solution 2: Téléchargement manuel**
```bash
# Télécharger avec curl (Windows PowerShell)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/healthcare-dataset-stroke-data.csv" -OutFile "healthcare-dataset-stroke-data.csv"

# Ou avec wget (si installé)
wget https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/healthcare-dataset-stroke-data.csv
```

### Erreur: "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Erreur: Port déjà utilisé

**Solution:**
```bash
# Utiliser un autre port
streamlit run app.py --server.port 8502
```

### L'application ne s'ouvre pas dans le navigateur

**Solution:**
1. Ouvrez manuellement: http://localhost:8501
2. Ou avec autre port: http://localhost:8502

---

## ⚙️ Configuration Avancée

### Modifier le Port

Éditez `.streamlit/config.toml`:
```toml
[server]
port = 8501
```

### Désactiver le Rerun Automatique

```toml
[server]
runOnSave = false
```

### Thème Personnalisé

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

---

## 📊 Dataset - Informations

### Source:
- **Nom**: Healthcare Dataset Stroke Data
- **Origine**: Kaggle (fedesoriano)
- **URL**: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### Caractéristiques:
- **Patients**: 5,110
- **Variables**: 11 features + 1 target
- **Classes**: Déséquilibré (4.87% stroke)

### Variables:
1. **id** - Identifiant unique
2. **gender** - Genre (Male/Female/Other)
3. **age** - Âge (0-120 ans)
4. **hypertension** - Hypertension (0/1)
5. **heart_disease** - Maladie cardiaque (0/1)
6. **ever_married** - Statut marital (Yes/No)
7. **work_type** - Type de travail (Private/Self-employed/Govt_job/children/Never_worked)
8. **Residence_type** - Type résidence (Urban/Rural)
9. **avg_glucose_level** - Niveau glucose moyen (mg/dL)
10. **bmi** - Indice masse corporelle (kg/m²)
11. **smoking_status** - Statut fumeur (formerly smoked/never smoked/smokes/Unknown)
12. **stroke** - Variable cible (0/1)

### Valeurs Manquantes:
- **bmi**: 201 valeurs (3.93%)
- **smoking_status**: 1,544 "Unknown" (30.21%)

---

## 🎯 Utilisation Recommandée

### Workflow Séquentiel:

1. **Étape 0**: Recherche Scientifique
   - Lisez les 5 articles de référence
   - Comprenez les méthodologies

2. **Étape 1**: Exploration des Données
   - Identifiez les valeurs manquantes
   - Analysez les statistiques de base

3. **Étape 2**: Analyse Univariée
   - Visualisez chaque variable
   - Comprenez les distributions

4. **Étape 3**: Analyse Bivariée
   - Étudiez les corrélations
   - Identifiez les relations

5. **Étape 4**: Prétraitement ✅
   - **OBLIGATOIRE** avant équilibrage
   - Imputez BMI (médiane recommandée)
   - Gérez smoking_status (mode ou garder Unknown)

6. **Étape 5**: Transformations
   - Normalisez les variables numériques
   - Encodez les variables catégorielles

7. **Étape 6**: Équilibrage ⚖️
   - **OBLIGATOIRE** avant ML
   - Utilisez **ADASYN** (recommandé)
   - Visualisez l'impact

8. **Étape 7a/b/c**: ACP/ACM/AFD
   - Analyses multivariées sur données équilibrées

9. **Étape 8**: Modèles ML 🤖
   - Entraînez **XGBoost** (recommandé)
   - Ou comparez tous les modèles

10. **Étape 9**: Prédiction Temps Réel 🔮
    - Testez avec des cas réels
    - Interprétez les résultats

---

## 📞 Support

### Problèmes Techniques:
- Vérifiez que toutes les dépendances sont installées
- Vérifiez que le dataset est présent
- Consultez les erreurs dans le terminal

### Questions Projet:
- Consultez `RESUME_AMELIORATIONS_FINALES.md`
- Lisez la documentation dans l'application (expanders 📚)

### Contact:
**MOAD MOUSSAOUI**  
📧 [votre_email@domaine.com]  
🔗 [LinkedIn]  

---

## 📜 Licence

Ce projet est à visée **éducative et académique**.

⚠️ **Disclaimer Médical**: Ne remplace PAS un diagnostic professionnel.

---

## 🎉 Bon Développement!

**Application prête à l'emploi - Suivez les étapes dans l'ordre!**

*Créé avec ❤️ par MOAD MOUSSAOUI - Génie Informatique 2025*
