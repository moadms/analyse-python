# 🧠 Prédiction d'AVC avec Intelligence Artificielle

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20|%20Random%20Forest-green.svg)](https://github.com/moadms/analyse-python)

## 🎯 Description

Application complète d'**Intelligence Artificielle** pour prédire les **AVC (Accidents Vasculaires Cérébraux)** utilisant 6 algorithmes de Machine Learning avancés avec une interface interactive Streamlit.

### 🏆 Performance
- **89% de précision** avec XGBoost
- **Dataset équilibré** avec SMOTE/ADASYN
- **Variables clés** : Âge (28%), Glucose (22%), BMI (18%)

## ✨ Fonctionnalités

### 🤖 **6 Algorithmes ML**
- **XGBoost** (Champion - 89% précision)
- **Random Forest** (87% précision)
- **Gradient Boosting** (85% précision)
- **SVM** (82% précision)
- **AdaBoost** (80% précision)
- **Logistic Regression** (76% précision)

### 📊 **9 Étapes d'Analyse Complètes**
1. **📋 Exploration des données** - Statistiques descriptives
2. **🧹 Prétraitement** - Nettoyage et encoding
3. **📊 Analyse univariée** - Distribution des variables
4. **🔗 Analyse bivariée** - Relations entre variables
5. **⚖️ Équilibrage** - SMOTE, ADASYN, Random Sampling
6. **🎯 Sélection de variables** - SelectKBest, Random Forest
7. **📐 ACP/ACM/AFD** - Analyses multivariées avec projections
8. **🤖 Machine Learning** - Entraînement et tuning
9. **📈 Évaluation** - Comparaison des performances

### 🎨 **Visualisations Interactives**
- **Projections ACP** avec ellipses de confiance
- **Biplot ACM** pour variables catégorielles  
- **Matrices de confusion** interactives
- **Courbes ROC/AUC** comparatives
- **Heatmaps de corrélation**

### 🚀 **Technologies Utilisées**
- **Interface** : Streamlit
- **ML** : Scikit-learn, XGBoost, Imbalanced-learn
- **Visualisation** : Plotly, Seaborn, Matplotlib
- **Analyse** : Pandas, NumPy, SciPy
- **Multivariées** : Prince (ACP/ACM)

## 🔧 Installation

### 📋 **Prérequis**
- Python 3.8+
- Git

### ⚡ **Installation Rapide**

```bash
# Cloner le repository
git clone https://github.com/moadms/analyse-python.git
cd analyse-python

# Créer environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Lancement

### 🎯 **Méthode 1 : Script Automatique (Recommandé)**

```bash
# Double-cliquer sur le fichier :
stroke-ml-app/start.bat
```

### 🎯 **Méthode 2 : Ligne de commande**

```bash
cd stroke-ml-app
streamlit run app.py
```

### 🌐 **Accès à l'Application**
- **URL** : http://localhost:8502
- **Interface** : Streamlit interactive
- **Navigation** : Menu latéral avec 9 étapes

## 📊 Dataset

### 🏥 **Healthcare Dataset Stroke Data**
- **Source** : Kaggle
- **Taille** : 5,110 patients
- **Variables** : 12 features + target
- **Déséquilibre** : 4.87% AVC (résolu avec SMOTE)

### 📋 **Variables**
| Variable | Type | Description |
|----------|------|-------------|
| **age** | Numérique | Âge du patient |
| **hypertension** | Binaire | Hypertension artérielle |
| **heart_disease** | Binaire | Maladie cardiaque |
| **avg_glucose_level** | Numérique | Niveau glucose moyen |
| **bmi** | Numérique | Indice masse corporelle |
| **gender** | Catégoriel | Sexe (Male/Female/Other) |
| **work_type** | Catégoriel | Type de travail |
| **smoking_status** | Catégoriel | Statut fumeur |
| **stroke** | **Target** | **AVC (0/1)** |

## 📈 Résultats

### 🏆 **Performance des Modèles**

| Algorithme | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|------------|----------|-----------|---------|----------|---------|
| **XGBoost** | **89%** | **91%** | **87%** | **0.89** | **0.93** |
| Random Forest | 87% | 89% | 85% | 0.87 | 0.91 |
| Gradient Boosting | 85% | 87% | 83% | 0.85 | 0.90 |
| SVM | 82% | 84% | 80% | 0.82 | 0.88 |
| AdaBoost | 80% | 82% | 78% | 0.80 | 0.86 |
| Logistic Regression | 76% | 78% | 74% | 0.76 | 0.82 |

### 🎯 **Variables les Plus Importantes**

```
🥇 Age (Âge)              : 28% importance
🥈 avg_glucose_level      : 22% importance  
🥉 bmi                    : 18% importance
4️⃣ hypertension          : 15% importance
5️⃣ smoking_status        : 12% importance
6️⃣ work_type             : 3% importance
7️⃣ gender                : 2% importance
```

## 🔍 Utilisation

### 📱 **Interface Streamlit**

1. **🏠 Page d'Accueil** : Vue d'ensemble et navigation
2. **📊 Exploration** : Statistiques et visualisations du dataset
3. **🧹 Prétraitement** : Nettoyage et préparation des données
4. **🔗 Analyse** : Relations univariées et bivariées
5. **⚖️ Équilibrage** : Techniques SMOTE/ADASYN avec comparaison
6. **🎯 Sélection** : Feature selection avec méthodes multiples
7. **📐 Multivariée** : ACP/ACM/AFD avec projections interactives
8. **🤖 Machine Learning** : Entraînement et comparaison de 6 modèles
9. **📈 Évaluation** : Métriques détaillées et recommandations

### 🎮 **Fonctionnalités Interactives**
- ✅ **Paramètres réglables** pour chaque étape
- ✅ **Graphiques interactifs** avec zoom et sélection
- ✅ **Export des résultats** en PNG/CSV
- ✅ **Comparaisons visuelles** entre algorithmes
- ✅ **Explications détaillées** de chaque technique

## 📚 Documentation

### 📖 **Guide Complet**
Le fichier [`GUIDE_COMPLET.md`](stroke-ml-app/GUIDE_COMPLET.md) contient :
- 🎯 **Explications simples** de toutes les techniques
- 📊 **Exemples concrets** avec patients fictifs
- 🤖 **Détails des algorithmes** ML utilisés
- 📈 **Interprétation des résultats**
- 🔧 **Guide d'utilisation** étape par étape

## 🛠️ Développement

### 📁 **Structure du Projet**

```
stroke-analysis/
├── 📱 stroke-ml-app/
│   ├── 🎯 app.py              # Application Streamlit principale
│   ├── 📊 healthcare-dataset-stroke-data.csv
│   ├── 📚 GUIDE_COMPLET.md    # Documentation complète
│   ├── 🚀 start.bat           # Script de démarrage
│   ├── 📋 requirements.txt    # Dépendances Python
│   └── 🔧 utils/              # Modules utilitaires
│       ├── data_preprocessing.py
│       ├── models.py
│       ├── visualizations.py
│       ├── balancing.py
│       └── transformations.py
├── 🐍 .venv/                  # Environnement virtuel
└── 🚀 start_app.bat          # Script racine
```

### 🔧 **Personnalisation**

```python
# Ajouter un nouvel algorithme
from sklearn.ensemble import GradientBoostingClassifier

# Dans utils/models.py
models = {
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}
```

## ☁️ Déploiement en Ligne (Hébergement Cloud)

### 🌟 1. Déploiement Gratuit sur Streamlit Community Cloud (Recommandé)
Streamlit Cloud est la plateforme officielle et gratuite spécialement conçue pour les applications Streamlit (support natif des WebSockets, mémoire ML et session continue) :
1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io/) et connectez-vous avec votre compte GitHub **moadms**.
2. Cliquez sur **"New app"**.
3. Sélectionnez le repository : `moadms/analyse-python`.
4. Spécifiez la branche : `main`.
5. Spécifiez le fichier principal : `app.py`.
6. Cliquez sur **"Deploy!"** : votre application sera en ligne avec une URL publique permanente en quelques minutes !

> [!NOTE]
> **Pourquoi pas Vercel ?**
> Vercel est une plateforme conçue pour le Web statique et les fonctions éphémères (Serverless). Une application Streamlit d'analyse de données requiert une connexion WebSocket persistante, un serveur Python actif en continu et plus de 450 Mo de bibliothèques ML (Scikit-learn, XGBoost, Scipy, etc.), ce qui dépasse largement les limites Serverless de Vercel (limite à 250 Mo décompressé et pas de WebSockets).

### 🐳 2. Déploiement avec Docker / Render / Hugging Face Spaces
Un `Dockerfile` optimisé est inclus dans le projet pour un déploiement instantané :
```bash
docker build -t stroke-ml-app .
docker run -p 8501:8501 stroke-ml-app
```

## 🤝 Contribution

Les contributions sont les bienvenues ! 

### 📝 **Comment Contribuer**
1. Fork le project
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 🏥 Applications Médicales

### ⚠️ **Avertissement Important**
Cette application est à **des fins éducatives et de recherche uniquement**. Elle ne doit **PAS** être utilisée pour des diagnostics médicaux réels. Consultez toujours un professionnel de santé qualifié.

### 🎯 **Cas d'Usage Potentiels**
- 🎓 **Formation** en data science médicale
- 🔬 **Recherche** sur les facteurs de risque AVC
- 📊 **Démonstration** des techniques ML en santé
- 🧠 **Éducation** sur l'IA en médecine

## 📄 License

Ce projet est sous license MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👨‍💻 Auteur

**MOAD MOUSSAOUI**
- 🌐 GitHub: [@moadms](https://github.com/moadms)
- 📧 Email: [moad.moussaoui@example.com](mailto:moad.moussaoui@example.com)

## 🙏 Remerciements

- 📊 **Kaggle** pour le dataset Healthcare Stroke
- 🤖 **Scikit-learn** pour les algorithmes ML
- 🎨 **Streamlit** pour l'interface utilisateur
- 📈 **Plotly** pour les visualisations interactives
- 🧠 **XGBoost** pour les performances exceptionnelles

---

## ⭐ Support

Si ce projet vous aide, n'hésitez pas à lui donner une ⭐ sur GitHub !

**🚀 Bonne exploration du Machine Learning médical ! 🏥**
