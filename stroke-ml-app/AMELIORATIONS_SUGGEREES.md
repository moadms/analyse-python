# 🎯 Plan d'Amélioration du Projet Stroke Analysis

**Date**: 27 Décembre 2025  
**Auteur**: MOAD MOUSSAOUI  
**Projet**: Analyse Prédictive des AVC - 4ème Année Génie Informatique

---

## 📊 ÉVALUATION GLOBALE

### Points Forts ✅
- ✅ Structure modulaire (utils/)
- ✅ 13 étapes bien définies
- ✅ Documentation scientifique (5 articles)
- ✅ Visualisations interactives (Plotly)
- ✅ Workflow séquentiel (session_state)
- ✅ Équilibrage complet (10 techniques)
- ✅ 6 algorithmes ML (dont XGBoost)
- ✅ Interface Streamlit professionnelle

### Score Global: **8.5/10** 🏆

---

## 🚨 PROBLÈMES CRITIQUES IDENTIFIÉS

### 1. CHRONOLOGIE INCORRECTE 🔴 **CRITIQUE**

**Problème**: Étape 5 (Transformations) est dans le code APRÈS Étapes 7a/7b/7c (ACP/ACM/AFD)

**Ordre actuel**:
```
app.py lignes:
- 2226: Étape 4 Prétraitement
- 2615: Étape 7a ACP        ← Utilise données non transformées!
- 2861: Étape 7b ACM        ← Utilise données non transformées!
- 3012: Étape 7c AFD        ← Utilise données non transformées!
- 3144: Étape 5 Transformations  ← Arrive trop tard!
- 3629: Étape 6 Équilibrage
- 4111: Étape 8 ML
```

**Impact**: 
- ACP/ACM/AFD travaillent sur données non normalisées
- Résultats biaisés
- Workflow illogique

**Solution**: Réorganiser code pour mettre Étape 5 à sa vraie place (ligne ~2600)

---

### 2. FEATURE ENGINEERING ABSENT 🔴 **CRITIQUE**

**Problème**: Aucune création de nouvelles features

**Impact sur performance**:
- XGBoost peut gagner +2-3% accuracy avec features engineered
- Variables catégorielles binaires améliorent interprétabilité

**Features à ajouter**:

#### A. Catégorisation BMI
```python
def create_bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'
```

#### B. Groupes d'âge
```python
def create_age_group(age):
    if age < 30: return 'Young'
    elif age < 50: return 'Adult'
    elif age < 70: return 'Senior'
    else: return 'Elderly'
```

#### C. Catégories glucose
```python
def create_glucose_category(glucose):
    if glucose < 100: return 'Normal'
    elif glucose < 126: return 'Prediabetic'
    else: return 'Diabetic'
```

#### D. Score de risque composite
```python
def create_risk_score(row):
    score = 0
    if row['age'] > 60: score += 2
    if row['hypertension'] == 1: score += 1
    if row['heart_disease'] == 1: score += 2
    if row['avg_glucose_level'] > 140: score += 1
    if row['bmi'] > 30: score += 1
    return score  # 0-7
```

#### E. Interactions features
```python
df['age_bmi_interaction'] = df['age'] * df['bmi']
df['glucose_bmi_interaction'] = df['avg_glucose_level'] * df['bmi']
df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
```

**Performance attendue**: +2-4% accuracy avec feature engineering

---

### 3. VALIDATION CROISÉE ABSENTE 🟡 **IMPORTANT**

**Problème**: Seulement train/test split simple (80/20)

**Impact**:
- Résultats peuvent varier selon le split aléatoire
- Pas de robustesse vérifiée
- Overfitting non détecté

**Solution**: K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Validation croisée stratifiée 5-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Évaluation
scores_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
scores_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
scores_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')

# Affichage
st.write(f"F1-Score: {scores_f1.mean():.2%} ± {scores_f1.std():.2%}")
st.write(f"Recall: {scores_recall.mean():.2%} ± {scores_recall.std():.2%}")
st.write(f"Precision: {scores_precision.mean():.2%} ± {scores_precision.std():.2%}")

# Graphique
fig = px.box(pd.DataFrame({
    'F1': scores_f1, 
    'Recall': scores_recall, 
    'Precision': scores_precision
}), title='Distribution Métriques (5-Fold CV)')
st.plotly_chart(fig)
```

---

### 4. EXPLAINABILITY (SHAP) ABSENTE 🟡 **IMPORTANT**

**Problème**: Aucune explication des prédictions individuelles

**Impact**:
- Médecins ne peuvent pas faire confiance à la "boîte noire"
- Impossible de valider cliniquement
- Non conforme aux standards médicaux (explainability requise)

**Solution**: Ajouter SHAP et LIME

#### A. Installation
```bash
pip install shap lime
```

#### B. SHAP pour XGBoost
```python
import shap

# Créer explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (importance globale)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(fig)

# Force plot (explication individuelle)
st.subheader("Explication d'une Prédiction Individuelle")
idx = st.slider("Sélectionner patient", 0, len(X_test)-1, 0)

fig = shap.force_plot(
    explainer.expected_value, 
    shap_values[idx], 
    X_test.iloc[idx],
    matplotlib=True,
    show=False
)
st.pyplot(fig)
```

#### C. LIME (alternative)
```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['No Stroke', 'Stroke'],
    mode='classification'
)

# Explication
exp = explainer.explain_instance(
    X_test.iloc[idx].values, 
    model.predict_proba
)

# Visualisation
fig = exp.as_pyplot_figure()
st.pyplot(fig)
```

---

### 5. COMPARAISON ÉQUILIBRAGE VISUELLE MANQUANTE 🟢 **NICE TO HAVE**

**Problème**: Vous testez 10 techniques mais pas de dashboard visuel

**Solution**: Graphique radar comparatif

```python
import plotly.graph_objects as go

# Après avoir testé toutes les techniques
techniques = ['SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', ...]
metrics = {
    'F1-Score': [0.92, 0.94, 0.91, 0.89, ...],
    'Recall': [0.91, 0.93, 0.90, 0.88, ...],
    'Precision': [0.93, 0.95, 0.92, 0.90, ...],
    'Training Time': [2.1, 2.5, 2.3, 3.1, ...]  # Normalisé 0-1
}

# Radar chart
fig = go.Figure()

for tech in techniques[:5]:  # Top 5
    idx = techniques.index(tech)
    fig.add_trace(go.Scatterpolar(
        r=[metrics['F1-Score'][idx], 
           metrics['Recall'][idx],
           metrics['Precision'][idx],
           1 - metrics['Training Time'][idx]/max(metrics['Training Time'])],
        theta=['F1', 'Recall', 'Precision', 'Speed'],
        fill='toself',
        name=tech
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title='Comparaison Techniques d\'Équilibrage'
)

st.plotly_chart(fig)

# Tableau comparatif
df_comparison = pd.DataFrame({
    'Technique': techniques,
    'F1-Score': metrics['F1-Score'],
    'Recall': metrics['Recall'],
    'Precision': metrics['Precision'],
    'Time (s)': metrics['Training Time']
}).sort_values('F1-Score', ascending=False)

st.dataframe(df_comparison.style.highlight_max(axis=0))
```

---

### 6. MÉTRIQUES AVANCÉES MANQUANTES 🟢 **NICE TO HAVE**

**Manque dans Étape 8**:

#### A. Precision-Recall Curve
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recall, y=precision,
    mode='lines',
    name=f'AP = {ap_score:.3f}'
))

fig.update_layout(
    title='Precision-Recall Curve',
    xaxis_title='Recall',
    yaxis_title='Precision'
)
st.plotly_chart(fig)
```

#### B. Learning Curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    cv=5,
    scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=train_sizes,
    y=train_scores.mean(axis=1),
    name='Train F1',
    mode='lines+markers'
))
fig.add_trace(go.Scatter(
    x=train_sizes,
    y=val_scores.mean(axis=1),
    name='Validation F1',
    mode='lines+markers'
))

fig.update_layout(
    title='Learning Curves',
    xaxis_title='Training Size',
    yaxis_title='F1-Score'
)
st.plotly_chart(fig)
```

#### C. Calibration Plot
```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=prob_pred, y=prob_true,
    mode='lines+markers',
    name='Calibration'
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Perfect Calibration',
    line=dict(dash='dash', color='red')
))

fig.update_layout(
    title='Calibration Plot',
    xaxis_title='Predicted Probability',
    yaxis_title='True Probability'
)
st.plotly_chart(fig)
```

---

### 7. TESTS STATISTIQUES ABSENTS 🟢 **NICE TO HAVE**

**À ajouter dans Étape 3**:

```python
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

st.subheader("Tests Statistiques")

# Chi² pour variables catégorielles vs stroke
categorical_vars = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                   'work_type', 'Residence_type', 'smoking_status']

chi2_results = []
for var in categorical_vars:
    contingency_table = pd.crosstab(df[var], df['stroke'])
    chi2, pvalue, dof, expected = chi2_contingency(contingency_table)
    chi2_results.append({
        'Variable': var,
        'Chi²': chi2,
        'p-value': pvalue,
        'Significatif': '✅' if pvalue < 0.05 else '❌'
    })

df_chi2 = pd.DataFrame(chi2_results)
st.dataframe(df_chi2.style.format({'Chi²': '{:.2f}', 'p-value': '{:.4f}'}))

st.info("""
**Interprétation**:
- **p-value < 0.05**: Variable est **significativement** associée au stroke
- **p-value ≥ 0.05**: Pas d'association significative
""")

# T-test pour variables numériques
numerical_vars = ['age', 'avg_glucose_level', 'bmi']

ttest_results = []
for var in numerical_vars:
    no_stroke = df[df['stroke'] == 0][var].dropna()
    stroke = df[df['stroke'] == 1][var].dropna()
    t_stat, pvalue = ttest_ind(stroke, no_stroke)
    
    ttest_results.append({
        'Variable': var,
        't-statistic': t_stat,
        'p-value': pvalue,
        'Différence Moyenne': stroke.mean() - no_stroke.mean(),
        'Significatif': '✅' if pvalue < 0.05 else '❌'
    })

df_ttest = pd.DataFrame(ttest_results)
st.dataframe(df_ttest.style.format({
    't-statistic': '{:.2f}', 
    'p-value': '{:.4f}',
    'Différence Moyenne': '{:.2f}'
}))
```

---

### 8. SAUVEGARDE MODÈLE & EXPORT ABSENTS 🟢 **NICE TO HAVE**

**À ajouter dans Étape 8**:

```python
import joblib
import io
from datetime import datetime

st.subheader("💾 Sauvegarde du Modèle")

# Bouton sauvegarde
if st.button("Télécharger Modèle Entraîné (.pkl)"):
    # Sauvegarder modèle + metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'balancing_method': st.session_state.workflow_data['balancing_method'],
        'metrics': {
            'f1_score': f1,
            'recall': recall,
            'precision': precision,
            'accuracy': accuracy
        },
        'date_training': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Serializer
    buffer = io.BytesIO()
    joblib.dump(model_data, buffer)
    buffer.seek(0)
    
    st.download_button(
        label="📥 Télécharger stroke_model.pkl",
        data=buffer,
        file_name=f"stroke_model_{model_name}_{datetime.now().strftime('%Y%m%d')}.pkl",
        mime="application/octet-stream"
    )

st.subheader("📊 Export Résultats")

# Export résultats CSV
results_df = pd.DataFrame({
    'Patient_ID': range(len(y_test)),
    'True_Label': y_test.values,
    'Predicted_Label': y_pred,
    'Probability_Stroke': y_pred_proba
})

csv = results_df.to_csv(index=False)
st.download_button(
    label="📥 Télécharger Résultats (CSV)",
    data=csv,
    file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)
```

---

## 📋 PLAN D'ACTION RECOMMANDÉ

### Phase 1: Corrections Critiques (1-2 jours)

1. ✅ **Réorganiser ordre étapes** (déplacer Étape 5 avant 7a/7b/7c)
2. ✅ **Ajouter Feature Engineering** (Étape 4b)
3. ✅ **Ajouter Cross-Validation** (Étape 8)

### Phase 2: Améliorations Importantes (3-4 jours)

4. ✅ **Ajouter SHAP/LIME** (nouvelle Étape 9)
5. ✅ **Dashboard comparaison équilibrage** (améliorer Étape 6)
6. ✅ **Métriques avancées** (Precision-Recall, Learning Curves)

### Phase 3: Finitions (1-2 jours)

7. ✅ **Tests statistiques** (améliorer Étape 3)
8. ✅ **Export & Sauvegarde** (Étape 8)
9. ✅ **Rapport PDF** (Étape 10 nouvelle)

---

## 🎯 STRUCTURE FINALE RECOMMANDÉE

```
🏠 Accueil
📚 Étape 0: Recherche Scientifique
📋 Étape 1: Exploration
📊 Étape 2: Analyse Univariée
🔗 Étape 3: Analyse Bivariée
📈 Étape 3b: Tests Statistiques (NOUVEAU)
🧹 Étape 4: Prétraitement
🎨 Étape 4b: Feature Engineering (NOUVEAU)
🔧 Étape 5: Transformations & Normalisation (DÉPLACÉ)
⚖️ Étape 6: Équilibrage (+ Dashboard Comparatif)
🎯 Étape 7a: ACP
📐 Étape 7b: ACM
🎲 Étape 7c: AFD
🤖 Étape 8: Modèles ML (+ CV, Métriques Avancées)
💡 Étape 9: Explainability (SHAP/LIME) (NOUVEAU)
🔮 Étape 10: Prédiction en Temps Réel
📄 Étape 11: Rapport & Export (NOUVEAU)
```

---

## 📊 SCORE FINAL ATTENDU

### Avant Améliorations: **8.5/10**
### Après Améliorations: **9.8/10** 🏆

**Détails**:
- Structure & Organisation: 9/10 → **10/10** (ordre corrigé)
- Feature Engineering: 0/10 → **9/10** (ajouté)
- Validation: 6/10 → **10/10** (CV ajoutée)
- Explainability: 0/10 → **10/10** (SHAP ajouté)
- Visualisations: 8/10 → **10/10** (dashboards ajoutés)
- Export: 0/10 → **9/10** (sauvegarde modèle)

---

## 💡 CONSEILS FINAUX

### Pour Présentation Académique

1. **Insister sur**:
   - SHAP/LIME (explainability médicale)
   - Feature engineering (intelligence du preprocessing)
   - Cross-validation (robustesse scientifique)
   - Tests statistiques (rigueur)

2. **Préparer slides**:
   - Avant/Après feature engineering (+X% performance)
   - Comparaison 10 techniques équilibrage (radar chart)
   - SHAP summary plot (variables importantes)
   - Learning curves (pas d'overfitting)

3. **Anticiper questions**:
   - "Pourquoi ADASYN > SMOTE?" → Montrer radar chart
   - "Comment expliquer prédiction?" → Montrer SHAP force plot
   - "Modèle robuste?" → Montrer CV scores ± std
   - "Quelles features importantes?" → Montrer SHAP summary

### Pour Déploiement Production

1. **Sauvegarder**:
   - Modèle (.pkl)
   - Scaler (.pkl)
   - Feature names (.json)
   - Metadata (.json)

2. **API REST** (optionnel):
   ```python
   # FastAPI endpoint
   @app.post("/predict")
   def predict(patient: PatientData):
       # Charger modèle
       model = joblib.load('stroke_model.pkl')
       # Prédire
       prediction = model.predict(patient.features)
       return {"stroke_risk": float(prediction[0])}
   ```

---

**Bon courage pour les améliorations! 🚀**

*Ce document est un guide complet pour porter votre projet de 8.5/10 à 9.8/10*
