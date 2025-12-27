# 📖 GUIDE D'UTILISATION - Application Stroke ML

## 🚀 Démarrage Rapide

### 1. Lancement de l'Application

```bash
# Option 1: Double-clic sur start.bat
start.bat

# Option 2: Terminal PowerShell
cd "d:\Stroke analyse\stroke-ml-app"
& "D:\Stroke analyse\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
```

**URL**: http://localhost:8501

---

## 📋 WORKFLOW RECOMMANDÉ (Ordre des Étapes)

### 🏁 Workflow Minimal (20 minutes)

```
1. 📋 Étape 1: Exploration
   → Voir valeurs manquantes + déséquilibre

2. 🧹 Étape 4: Prétraitement
   → Imputer BMI (médiane)
   → Cliquer "APPLIQUER PRÉTRAITEMENT" ✅

3. ⚖️ Étape 6: Équilibrage
   → Choisir ADASYN (recommandé)
   → Cliquer "APPLIQUER" ✅

4. 🤖 Étape 8: Modèles ML
   → Mode "Modèle Individuel"
   → Choisir XGBoost ou Random Forest
   → Cliquer "ENTRAÎNER"
   → (Optionnel) Lancer Cross-Validation 5-Fold

5. 🔮 Prédiction en Temps Réel
   → Entrer données patient
   → Obtenir prédiction + recommandations

6. 📄 Génération de Rapports
   → Personnaliser sections
   → Télécharger Markdown/HTML
```

**Temps total**: ~20 minutes

---

### 🏆 Workflow Complet (60 minutes)

```
1. 🏠 Accueil
   → Lire intro + objectifs

2. 📚 Étape 0: Recherche Scientifique
   → Explorer 5 articles scientifiques
   → Comprendre contexte académique

3. 📋 Étape 1: Exploration
   → Statistiques dataset
   → Valeurs manquantes (BMI: 201, smoking_status: 1,544 'Unknown')
   → Distribution cible (4.87% stroke = TRÈS déséquilibré)

4. 📊 Étape 2: Analyse Univariée
   → Chaque variable individuellement
   → Q-Q plots (normalité)
   → Boxplots (outliers)

5. 🔗 Étape 3: Analyse Bivariée
   → Matrice corrélations
   → Violin plots stroke vs variables
   → Chi² tests indépendance

6. 🧹 Étape 4: Prétraitement ✅
   → Imputation BMI médiane
   → Gestion smoking_status 'Unknown'
   → Encodage catégorielles
   → ✅ CLIQUER "APPLIQUER PRÉTRAITEMENT"

7. 🔧 Étape 5: Transformations (Optionnel)
   → Tester normalité (Q-Q plots)
   → Appliquer transformations si besoin (log, Box-Cox)
   → Normaliser (StandardScaler recommandé avant ACP/AFD)

8. ⚖️ Étape 6: Équilibrage ✅
   Tab "📊 Visualiser Impact":
   → ✅ LANCER "Comparaison Complète" (~1 min)
   → Voir recommandation empirique
   
   Tab "🎯 Méthodes Over-sampling":
   → Choisir ADASYN (ou technique recommandée)
   → ✅ CLIQUER "APPLIQUER"
   
   Tab "📊 Visualiser Impact":
   → Vérifier distributions avant/après

9. 🎯 Étape 7a: ACP
   → Réduction dimensionnelle
   → Variance expliquée
   → Biplot (variables + individus)

10. 📐 Étape 7b: ACM
    → Analyse variables catégorielles
    → Projection modalités
    → Relations entre catégories

11. 🎲 Étape 7c: AFD
    → Axes discriminants (séparation classes)
    → Coefficients discriminants (variables importantes)
    → Qualité séparation (centroïdes)

12. 🤖 Étape 8: Modèles ML
    Mode "Comparaison Tous Modèles":
    → Entraîne les 6 algorithmes
    → Compare F1-Score, Recall, Precision
    → Identifie meilleur modèle
    
    Mode "Modèle Individuel":
    → Choisir meilleur modèle (ex: XGBoost)
    → Entraîner
    → ✅ LANCER "Cross-Validation 5-Fold"
    → Vérifier stabilité (std < 3% = excellent)
    → Voir feature importance

13. 🔮 Prédiction en Temps Réel
    → Entrer données patient fictif
    → Obtenir probabilité + niveau risque
    → Voir facteurs de risque identifiés
    → Lire recommandations médicales

14. 📄 Génération de Rapports
    → Sélectionner sections (6 disponibles)
    → Générer rapport
    → Télécharger Markdown + HTML
    → Export métriques Excel
    → Export données CSV
```

**Temps total**: ~60 minutes (analyse complète)

---

## 🎯 CAS D'USAGE SPÉCIFIQUES

### 1. Présentation Académique (Prof)

```
Objectif: Démontrer méthodologie ML complète

Workflow:
1. 📋 Étape 1: Montrer déséquilibre (4.87% stroke)
2. 🧹 Étape 4: Preprocessing (imputation BMI)
3. ⚖️ Étape 6: Comparison Dashboard
   → Montrer comparaison 10 techniques
   → Justifier choix ADASYN empiriquement
4. 🤖 Étape 8: Entraîner tous modèles
   → Tableau comparatif
   → Cross-Validation meilleur modèle
   → Feature importance
5. 📄 Rapports: Générer rapport complet

Durée: 15-20 minutes
Points clés:
- ✅ Méthodologie scientifique rigoureuse
- ✅ Validation croisée (robustesse)
- ✅ Comparaison empirique (justification choix)
- ✅ Documentation professionnelle
```

---

### 2. Analyse Rapide d'un Patient

```
Objectif: Prédire risque AVC patient spécifique

Workflow:
1. Vérifier que modèles sont entraînés (Étape 8)
2. 🔮 Prédiction en Temps Réel
3. Remplir formulaire patient:
   - Âge, genre, hypertension, etc.
4. Cliquer "PRÉDIRE RISQUE AVC"
5. Lire résultat:
   - Probabilité % + niveau risque (🔴🟡🟢)
   - Facteurs de risque identifiés
   - Recommandations médicales

Durée: 2 minutes
⚠️ DISCLAIMER: Pas un diagnostic médical!
```

---

### 3. Recherche Best Balancing Technique

```
Objectif: Identifier meilleure technique équilibrage pour CE dataset

Workflow:
1. 🧹 Étape 4: Prétraitement (obligatoire)
2. ⚖️ Étape 6 → Tab "📊 Visualiser Impact"
3. Cliquer "🚀 Lancer Comparaison Complète"
4. Attendre ~1 minute (10 techniques testées)
5. Analyser résultats:
   - Radar chart (top 5)
   - Tableau complet (10 techniques)
   - Graphique F1-Score
   - Performance vs Vitesse
6. Lire recommandation automatique
7. Appliquer technique gagnante (tab 1)

Durée: 5 minutes
Résultat: Choix data-driven, justifié empiriquement
```

---

### 4. Export Documentation Projet

```
Objectif: Générer documentation complète pour soumission

Workflow:
1. Compléter workflow minimal (étapes 1,4,6,8)
2. 📄 Génération de Rapports
3. Tab "📊 Rapport Complet":
   - Cocher toutes sections ✅
   - Générer rapport
   - Télécharger Markdown (.md)
   - Télécharger HTML (.html)
4. Tab "📈 Métriques ML":
   - Télécharger Excel (métriques + données)
5. Tab "📋 Données Export":
   - Télécharger CSV original
   - Télécharger CSV équilibré

Durée: 5 minutes
Fichiers générés:
- rapport_stroke_analysis_YYYYMMDD.md
- rapport_stroke_analysis_YYYYMMDD.html
- metriques_ml_YYYYMMDD.xlsx
- stroke_data_original_YYYYMMDD.csv
- stroke_data_balanced_YYYYMMDD.csv
```

---

## 🎓 CONSEILS POUR PRÉSENTATION ORALE

### Structure Recommandée (15 minutes)

**1. Introduction (2 min)**
- Problématique: AVC = 2ème cause mortalité mondiale
- Objectif: Prédire risque AVC via ML
- Dataset: 5,110 patients, 4.87% stroke (déséquilibré)

**2. Méthodologie (5 min)**

*Montrer Étape 1*:
- "Voici notre dataset: 201 valeurs manquantes BMI..."
- "Déséquilibre 1:19.5 → Problème majeur ML"

*Montrer Étape 4*:
- "Prétraitement: imputation médiane, encodage..."

*Montrer Étape 6*:
- "Comparaison 10 techniques équilibrage..."
- "ADASYN gagne empiriquement: F1=XX%, Recall=XX%"
- "Justification data-driven, pas arbitraire"

**3. Résultats ML (5 min)**

*Montrer Étape 8*:
- "6 algorithmes testés..."
- "Tableau comparatif: XGBoost meilleur (F1=XX%)"
- "Cross-Validation confirme robustesse (std=X.X%)"
- "Feature importance: âge, hypertension, glucose..."

**4. Application Pratique (2 min)**

*Montrer Prédiction*:
- "Interface temps réel pour médecins..."
- "Patient 65 ans, hypertendu → Risque 75% 🔴"
- "Recommandations personnalisées..."

**5. Conclusion (1 min)**
- Workflow complet implémenté
- Validation rigoureuse (CV, comparaison empirique)
- Production-ready (prédiction + rapports)
- Limitations: éducatif uniquement, pas diagnostic

---

### Slides Suggérées

1. **Titre**: Analyse Prédictive AVC par Machine Learning
2. **Contexte**: Statistiques AVC + importance prédiction
3. **Dataset**: 5,110 patients, 11 variables, 4.87% stroke
4. **Déséquilibre**: Graphique avant/après ADASYN
5. **Méthodologie**: Flowchart 13 étapes
6. **Comparaison Équilibrage**: Radar chart 10 techniques
7. **Résultats ML**: Tableau 6 modèles comparés
8. **Cross-Validation**: Boxplot + interprétation stabilité
9. **Feature Importance**: Top 5 variables prédictives
10. **Application**: Screenshot prédiction patient
11. **Rapports**: Screenshot rapport généré
12. **Conclusion**: Récap contributions + limitations

---

## ⚠️ ERREURS COMMUNES ET SOLUTIONS

### Erreur 1: "Prétraitement requis"
**Symptôme**: Message dans Étapes 6, 7, 8
**Cause**: Étape 4 pas complétée
**Solution**: 
1. Aller Étape 4
2. Cliquer "APPLIQUER PRÉTRAITEMENT"
3. Attendre message "✅ Données prétraitées sauvegardées"

---

### Erreur 2: "Aucun modèle entraîné"
**Symptôme**: Page Prédiction vide
**Cause**: Étape 8 pas complétée
**Solution**:
1. Aller Étape 8
2. Choisir modèle (ex: XGBoost)
3. Cliquer "ENTRAÎNER"
4. Attendre fin entraînement

---

### Erreur 3: Cross-Validation bloquée
**Symptôme**: Bouton CV désactivé
**Cause**: Modèle pas entraîné
**Solution**:
1. Entraîner modèle d'abord
2. Puis lancer CV

---

### Erreur 4: Rapports vides
**Symptôme**: Rapport généré sans contenu
**Cause**: Workflow pas complété
**Solution**:
1. Compléter au minimum:
   - Étape 4 (Prétraitement)
   - Étape 6 (Équilibrage)
   - Étape 8 (ML)
2. Cocher sections correspondantes
3. Régénérer rapport

---

## 📊 MÉTRIQUES INTERPRÉTATION

### F1-Score
- **> 0.85**: Excellent ✅
- **0.70-0.85**: Bon ℹ️
- **< 0.70**: Faible ⚠️

**Formule**: $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

---

### Recall (Sensibilité)
- **> 0.90**: Excellent ✅ (détecte 90%+ AVC)
- **0.70-0.90**: Bon ℹ️
- **< 0.70**: Dangereux ⚠️ (trop de faux négatifs)

**Important**: En médical, Recall PRIORITAIRE (ne pas manquer un AVC)

---

### Precision (Précision)
- **> 0.85**: Excellent ✅ (peu faux positifs)
- **0.70-0.85**: Acceptable ℹ️
- **< 0.70**: Trop fausses alertes ⚠️

---

### Cross-Validation Std
- **< 3%**: Modèle TRÈS STABLE ✅
- **3-5%**: Stable ℹ️
- **> 5%**: INSTABLE ⚠️ (overfitting probable)

---

## 🔧 DÉPANNAGE

### Application ne démarre pas
```powershell
# Vérifier environnement virtuel
cd "d:\Stroke analyse"
.venv\Scripts\Activate.ps1

# Vérifier packages
pip list | Select-String "streamlit|pandas|sklearn"

# Réinstaller si nécessaire
pip install -r stroke-ml-app/requirements.txt

# Relancer
cd stroke-ml-app
streamlit run app.py
```

---

### Port 8501 déjà utilisé
```powershell
# Utiliser port différent
streamlit run app.py --server.port 8502

# Ou tuer processus Python
Get-Process python | Stop-Process -Force
```

---

### Graphiques ne s'affichent pas
**Cause**: Plotly pas installé
**Solution**:
```powershell
pip install plotly>=5.17.0
```

---

## 📚 RESSOURCES SUPPLÉMENTAIRES

### Fichiers du Projet
- `app.py` (5,228 lignes): Application principale
- `utils/models.py`: 6 algorithmes ML
- `utils/balancing.py`: 10 techniques équilibrage
- `utils/visualizations.py`: Graphiques Plotly
- `AMELIORATIONS_FINALES.md`: Documentation complète
- `GUIDE_UTILISATION.md`: Ce guide

### Documentation Externe
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

## ✅ CHECKLIST DÉMONSTRATION

Avant présentation, vérifier:
- [ ] Application démarre (http://localhost:8501)
- [ ] Toutes pages accessibles
- [ ] Prétraitement fonctionne (Étape 4)
- [ ] Équilibrage fonctionne (Étape 6)
- [ ] Comparison Dashboard fonctionne (~1 min)
- [ ] ML entraîne modèles (Étape 8)
- [ ] Cross-Validation fonctionne
- [ ] Prédiction fonctionne (patient fictif préparé)
- [ ] Rapports génèrent et téléchargent
- [ ] Screenshots prêts pour slides

---

**Bon courage pour votre présentation! 🚀**

---

**Généré le**: 27 Décembre 2025  
**Par**: MOAD MOUSSAOUI - Projet 4ème Année Génie Informatique
