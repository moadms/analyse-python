"""
Application Streamlit d'Analyse de Stroke - Version Refactorisée
Architecture Modulaire et Lisible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import prince

# Import des modules utilitaires
from utils.data_preprocessing import (
    load_and_clean_data, preprocess_data, detect_outliers, 
    handle_outliers, get_basic_stats
)
from utils.transformations import DataTransformer, auto_transform
from utils.balancing import DataBalancer, compare_balancing_methods
from utils.visualizations import *
from utils.models import ModelTrainer

# Configuration de la page
st.set_page_config(
    page_title="Analyse Stroke ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e40af;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #3b82f6;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cache pour les données
@st.cache_data
def load_data():
    """Charge les données avec cache."""
    return load_and_clean_data('healthcare-dataset-stroke-data.csv')

# Fonction principale
def main():
    # Titre principal
    st.markdown('<h1 class="main-header">🏥 Analyse Prédictive des AVC (Stroke)</h1>', unsafe_allow_html=True)
    
    # Charger les données
    df = load_data()
    
    # Sidebar - Navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio(
        "Choisir une étape:",
        [
            "🏠 Accueil",
            "📋 Étape 1: Exploration des Données",
            "📊 Étape 2: Analyse Univariée",
            "🔗 Étape 3: Analyse Bivariée",
            "🎯 Étape 4a: PCA",
            "📐 Étape 4b: ACM",
            "🎲 Étape 4c: AFD (LDA)",
            "🔧 Étape 5: Transformations & Normalisation",
            "⚖️ Étape 6: Équilibrage des Données",
            "🤖 Étape 7: Modèles ML",
            "🔮 Prédiction en Temps Réel"
        ]
    )
    
    # Affichage des statistiques de base dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Statistiques")
    stats = get_basic_stats(df)
    st.sidebar.metric("Nombre de patients", stats['shape'][0])
    st.sidebar.metric("Nombre de variables", stats['shape'][1])
    stroke_rate = (df['stroke'].sum() / len(df)) * 100
    st.sidebar.metric("Taux de Stroke", f"{stroke_rate:.2f}%")
    
    # === PAGE: ACCUEIL ===
    if page == "🏠 Accueil":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Objectifs
            - Prédire les AVC
            - Identifier les facteurs de risque
            - Modèles ML performants
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Dataset
            - 5,110 patients
            - 11 variables
            - Déséquilibre: 4.87%
            """)
        
        with col3:
            st.markdown("""
            ### 🛠️ Technologies
            - Python, Streamlit
            - Scikit-learn
            - Plotly, Seaborn
            """)
        
        st.markdown("---")
        st.info("""
        ### 📚 Workflow Complet
        1. **Exploration**: Comprendre les données
        2. **Analyse Univariée**: Distribution de chaque variable
        3. **Analyse Bivariée**: Relations entre variables
        4. **Analyses Multivariées**: PCA, ACM, AFD
        5. **Transformations**: Log, Box-Cox, Normalisation
        6. **Équilibrage**: SMOTE, ADASYN, Tomek Links
        7. **ML Models**: Random Forest, Gradient Boosting, etc.
        8. **Prédiction**: Outil de prédiction en temps réel
        """)
    
    # === PAGE: EXPLORATION ===
    elif page == "📋 Étape 1: Exploration des Données":
        st.markdown('<h2 class="section-header">📋 Exploration des Données</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Aperçu", "🔍 Statistiques", "📈 Distribution Cible"])
        
        with tab1:
            st.subheader("Aperçu du Dataset")
            st.dataframe(df.head(100), use_container_width=True, height=400)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Types de données:**")
                st.dataframe(df.dtypes.to_frame('Type'), use_container_width=True)
            
            with col2:
                st.write("**Valeurs manquantes:**")
                missing = df.isnull().sum()
                st.dataframe(missing.to_frame('Manquantes'), use_container_width=True)
        
        with tab2:
            st.subheader("Statistiques Descriptives")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.write("**Variables Catégorielles:**")
            for col in stats['categorical_cols']:
                with st.expander(f"Distribution de {col}"):
                    counts = df[col].value_counts()
                    fig = px.pie(values=counts.values, names=counts.index, title=col)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Distribution de la Variable Cible (Stroke)")
            
            stroke_counts = df['stroke'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_class_distribution(df['stroke'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Patients sans Stroke", stroke_counts[0])
                st.metric("Patients avec Stroke", stroke_counts[1])
                ratio = stroke_counts[0] / stroke_counts[1]
                st.metric("Ratio Déséquilibre", f"1:{ratio:.1f}")
                
                st.warning(f"""
                **⚠️ Déséquilibre Important!**
                - Seulement {stroke_rate:.2f}% de cas positifs
                - Nécessite des techniques de rééquilibrage (SMOTE, etc.)
                """)
    
    # === PAGE: ANALYSE UNIVARIÉE ===
    elif page == "📊 Étape 2: Analyse Univariée":
        st.markdown('<h2 class="section-header">📊 Analyse Univariée</h2>', unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('stroke') if 'stroke' in numeric_cols else None
        
        selected_var = st.selectbox("Choisir une variable numérique", numeric_cols)
        
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Q-Q Plot", "📦 Boxplot"])
        
        with tab1:
            fig = create_distribution_plot(df, selected_var)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Moyenne", f"{df[selected_var].mean():.2f}")
            col2.metric("Médiane", f"{df[selected_var].median():.2f}")
            col3.metric("Écart-type", f"{df[selected_var].std():.2f}")
            
            skewness = df[selected_var].skew()
            if abs(skewness) > 1:
                st.warning(f"**Asymétrie forte détectée** (skew={skewness:.2f}). Transformation recommandée.")
            elif abs(skewness) > 0.5:
                st.info(f"**Asymétrie modérée** (skew={skewness:.2f}).")
        
        with tab2:
            fig = create_qq_plot(df, selected_var)
            st.plotly_chart(fig, use_container_width=True)
            st.info("Si les points suivent la ligne rouge, la distribution est proche de la normale.")
        
        with tab3:
            fig = create_boxplot(df, selected_var, by='stroke')
            st.plotly_chart(fig, use_container_width=True)
            
            outliers = detect_outliers(df, [selected_var])
            n_outliers = len(outliers[selected_var])
            if n_outliers > 0:
                st.warning(f"**{n_outliers} valeurs aberrantes détectées** ({n_outliers/len(df)*100:.1f}%)")
    
    # === PAGE: ANALYSE BIVARIÉE ===
    elif page == "🔗 Étape 3: Analyse Bivariée":
        st.markdown('<h2 class="section-header">🔗 Analyse Bivariée</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔥 Corrélations", "🎻 Violin Plots"])
        
        with tab1:
            st.subheader("Matrice de Corrélation")
            df_processed, _ = preprocess_data(df)
            
            method = st.radio("Méthode", ["pearson", "spearman"], horizontal=True)
            fig = create_correlation_heatmap(df_processed, method=method)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **💡 Interprétation:**
            - Corrélation > 0.7: Forte corrélation positive
            - Corrélation < -0.7: Forte corrélation négative
            - Proche de 0: Pas de corrélation linéaire
            """)
        
        with tab2:
            st.subheader("Distribution par Variable Cible")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.remove('stroke') if 'stroke' in numeric_cols else None
            
            selected_var = st.selectbox("Variable à analyser", numeric_cols)
            fig = create_violin_plot(df, x='stroke', y=selected_var, color='stroke')
            st.plotly_chart(fig, use_container_width=True)
    
    # === PAGE: PCA ===
    elif page == "🎯 Étape 4a: PCA":
        st.markdown('<h2 class="section-header">🎯 Analyse en Composantes Principales (PCA)</h2>', unsafe_allow_html=True)
        
        df_processed, _ = preprocess_data(df)
        X = df_processed.drop('stroke', axis=1)
        y = df_processed['stroke']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_components = st.slider("Nombre de composantes", 2, min(10, X.shape[1]), 3)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        tab1, tab2, tab3 = st.tabs(["📊 Variance Expliquée", "🎯 Projection 2D", "📐 Cercle de Corrélation"])
        
        with tab1:
            explained_var = pca.explained_variance_ratio_ * 100
            cum_var = np.cumsum(explained_var)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(x=[f'PC{i+1}' for i in range(n_components)], y=explained_var,
                           title='Variance Expliquée', labels={'x': 'Composante', 'y': '% Variance'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(x=[f'PC{i+1}' for i in range(n_components)], y=cum_var,
                            title='Variance Cumulée', markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"Les {n_components} premières composantes expliquent **{cum_var[-1]:.1f}%** de la variance totale.")
        
        with tab2:
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'stroke': y.values
            })
            
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='stroke',
                           title='Projection PCA (2D)',
                           color_discrete_sequence=['#4ade80', '#ef4444'],
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
            
            fig = go.Figure()
            
            for i, feature in enumerate(X.columns):
                fig.add_trace(go.Scatter(
                    x=[0, loadings[i, 0]],
                    y=[0, loadings[i, 1]],
                    mode='lines+text',
                    name=feature,
                    text=['', feature],
                    textposition='top center'
                ))
            
            fig.add_shape(type='circle', xref='x', yref='y',
                        x0=-1, y0=-1, x1=1, y1=1,
                        line_color='gray', line_dash='dash')
            
            fig.update_layout(title='Cercle des Corrélations', 
                            width=700, height=700)
            st.plotly_chart(fig, use_container_width=True)
    
    # === PAGE: ACM ===
    elif page == "📐 Étape 4b: ACM":
        st.markdown('<h2 class="section-header">📐 Analyse des Correspondances Multiples (ACM)</h2>', unsafe_allow_html=True)
        
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        df_cat = df[categorical_cols + ['stroke']].copy()
        
        st.info("**L'ACM** analyse les relations entre variables catégorielles.")
        
        try:
            mca = prince.MCA(n_components=3, random_state=42)
            mca_result = mca.fit_transform(df_cat[categorical_cols])
            
            tab1, tab2 = st.tabs(["🎯 Projection Individus", "📊 Modalités"])
            
            with tab1:
                mca_df = pd.DataFrame({
                    'Dim1': mca_result[0].values,
                    'Dim2': mca_result[1].values,
                    'stroke': df_cat['stroke'].values
                })
                
                fig = px.scatter(mca_df, x='Dim1', y='Dim2', color='stroke',
                               title='Projection ACM des Individus',
                               color_discrete_sequence=['#4ade80', '#ef4444'],
                               opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                column_coords = mca.column_coordinates(df_cat[categorical_cols])
                
                fig = go.Figure()
                
                for col in categorical_cols:
                    col_data = column_coords.loc[column_coords.index.str.startswith(f"{col}_")]
                    fig.add_trace(go.Scatter(
                        x=col_data[0],
                        y=col_data[1],
                        mode='markers+text',
                        name=col,
                        text=col_data.index.str.replace(f"{col}_", ""),
                        textposition='top center',
                        marker=dict(size=12)
                    ))
                
                fig.update_layout(title='Projection des Modalités (ACM)', width=900, height=700)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erreur ACM: {e}")
    
    # === PAGE: AFD ===
    elif page == "🎲 Étape 4c: AFD (LDA)":
        st.markdown('<h2 class="section-header">🎲 Analyse Factorielle Discriminante (LDA)</h2>', unsafe_allow_html=True)
        
        df_processed, _ = preprocess_data(df)
        X = df_processed.drop('stroke', axis=1)
        y = df_processed['stroke']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(X_scaled, y)
        
        tab1, tab2 = st.tabs(["📊 Distribution", "🔍 Coefficients"])
        
        with tab1:
            lda_df = pd.DataFrame({
                'LD1': X_lda.flatten(),
                'stroke': y.values
            })
            
            fig = go.Figure()
            
            for stroke_val, color, name in [(0, '#4ade80', 'Pas de Stroke'), (1, '#ef4444', 'Stroke')]:
                data = lda_df[lda_df['stroke'] == stroke_val]['LD1']
                fig.add_trace(go.Histogram(
                    x=data,
                    name=name,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=50
                ))
            
            fig.update_layout(title='Distribution sur l\'Axe Discriminant',
                            barmode='overlay', width=900, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            coefficients = pd.DataFrame({
                'Variable': X.columns,
                'Coefficient': lda.coef_[0],
                'Importance': np.abs(lda.coef_[0])
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(coefficients.head(10), y='Variable', x='Coefficient',
                       orientation='h', title='Top 10 Variables Discriminantes',
                       color='Coefficient', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(coefficients, use_container_width=True)
    
    # === PAGE: TRANSFORMATIONS & NORMALISATION ===
    elif page == "🔧 Étape 5: Transformations & Normalisation":
        st.markdown('<h2 class="section-header">🔧 Transformations & Normalisation</h2>', unsafe_allow_html=True)
        
        st.info("""
        **Objectif**: Traiter les valeurs aberrantes et normaliser les distributions asymétriques.
        
        **Techniques disponibles**:
        - Transformation Log (log(x+1))
        - Transformation Racine Carrée (√x)
        - Box-Cox (trouve le meilleur λ automatiquement)
        - Yeo-Johnson (fonctionne avec valeurs négatives)
        - Quantile Transform (vers distribution normale)
        - StandardScaler (z-score)
        - MinMaxScaler ([0, 1])
        - RobustScaler (robuste aux outliers)
        """)
        
        transformer = DataTransformer()
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Détection Outliers", "🔧 Transformations", "📊 Normalisation", "📈 Comparaisons"])
        
        with tab1:
            st.subheader("Détection des Valeurs Aberrantes")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.remove('stroke') if 'stroke' in numeric_cols else None
            
            method = st.radio("Méthode de détection", ["IQR (Interquartile Range)", "Z-Score"], horizontal=True)
            
            if method == "IQR (Interquartile Range)":
                outliers = detect_outliers(df, numeric_cols, method='iqr', threshold=1.5)
            else:
                outliers = detect_outliers(df, numeric_cols, method='zscore', threshold=3)
            
            outlier_summary = pd.DataFrame({
                'Variable': list(outliers.keys()),
                'Nombre Outliers': [len(v) for v in outliers.values()],
                'Pourcentage': [len(v)/len(df)*100 for v in outliers.values()]
            }).sort_values('Nombre Outliers', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(outlier_summary, use_container_width=True)
            
            with col2:
                fig = px.bar(outlier_summary, x='Variable', y='Pourcentage',
                           title='% Outliers par Variable',
                           color='Pourcentage', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            
            selected_var = st.selectbox("Visualiser les outliers pour:", numeric_cols)
            if selected_var in outliers:
                fig = create_outlier_visualization(df, selected_var, outliers[selected_var])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Transformations de Variables")
            
            # Calcul du skewness
            skewness_df = transformer.calculate_skewness(df, numeric_cols)
            st.write("**Asymétrie (Skewness) des Variables:**")
            st.dataframe(skewness_df.style.background_gradient(cmap='RdYlGn_r', axis=0), use_container_width=True)
            
            st.markdown("---")
            
            # Suggestions
            suggestions = transformer.suggest_transformations(df)
            st.write("**Transformations Suggérées:**")
            suggestions_df = pd.DataFrame.from_dict(suggestions, orient='index', columns=['Suggestion'])
            st.dataframe(suggestions_df, use_container_width=True)
            
            st.markdown("---")
            
            # Appliquer transformation
            st.subheader("Appliquer une Transformation")
            
            selected_var = st.selectbox("Choisir une variable:", numeric_cols, key='transform_var')
            transform_method = st.selectbox("Méthode de transformation:", 
                                          ['log', 'sqrt', 'boxcox', 'yeo-johnson', 'quantile'])
            
            if st.button("Appliquer Transformation"):
                if transform_method == 'log':
                    df_transformed = transformer.log_transform(df, [selected_var])
                elif transform_method == 'sqrt':
                    df_transformed = transformer.sqrt_transform(df, [selected_var])
                elif transform_method == 'boxcox':
                    df_transformed = transformer.box_cox_transform(df, [selected_var])
                elif transform_method == 'yeo-johnson':
                    df_transformed = transformer.yeo_johnson_transform(df, [selected_var])
                else:
                    df_transformed = transformer.quantile_transform(df, [selected_var])
                
                # Visualisation
                fig = create_transformation_comparison(df, df_transformed, selected_var, transform_method)
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparaison skewness
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skewness Avant", f"{df[selected_var].skew():.3f}")
                with col2:
                    transformed_col = f'{selected_var}_{transform_method}'
                    if transformed_col in df_transformed.columns:
                        st.metric("Skewness Après", f"{df_transformed[transformed_col].skew():.3f}")
        
        with tab3:
            st.subheader("Normalisation / Standardisation")
            
            scale_method = st.radio("Méthode:", 
                                  ["StandardScaler (Z-score)", "MinMaxScaler [0,1]", "RobustScaler (Median/IQR)"],
                                  horizontal=True)
            
            if st.button("Appliquer Normalisation"):
                df_numeric = df[numeric_cols].copy()
                
                if "StandardScaler" in scale_method:
                    df_scaled = transformer.standard_scale(df_numeric)
                    method_name = "StandardScaler"
                elif "MinMaxScaler" in scale_method:
                    df_scaled = transformer.minmax_scale(df_numeric)
                    method_name = "MinMaxScaler"
                else:
                    df_scaled = transformer.robust_scale(df_numeric)
                    method_name = "RobustScaler"
                
                # Afficher les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Données Originales (extrait):**")
                    st.dataframe(df_numeric.head(10), use_container_width=True)
                
                with col2:
                    st.write(f"**Données Normalisées ({method_name}):**")
                    st.dataframe(df_scaled.head(10), use_container_width=True)
                
                # Visualisation
                selected_var = st.selectbox("Visualiser pour:", numeric_cols, key='viz_scaled')
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df_numeric[selected_var], name='Original', opacity=0.7))
                fig.add_trace(go.Histogram(x=df_scaled[selected_var], name=f'Normalisé ({method_name})', opacity=0.7))
                fig.update_layout(title=f'Comparaison {selected_var}', barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Traitement Automatique des Outliers")
            
            treatment = st.radio("Méthode de traitement:", 
                               ["Winsorization (Capping 5%-95%)", "Suppression (IQR)"],
                               horizontal=True)
            
            if st.button("Traiter les Outliers"):
                if "Winsorization" in treatment:
                    df_clean = handle_outliers(df, numeric_cols, method='cap')
                    method_name = "Winsorization"
                else:
                    df_clean = handle_outliers(df, numeric_cols, method='remove')
                    method_name = "Suppression"
                
                st.success(f"**{method_name} appliquée!**")
                st.write(f"Taille avant: {len(df)} → Taille après: {len(df_clean)}")
                
                # Comparaison
                selected_var = st.selectbox("Comparer pour:", numeric_cols, key='compare_outlier')
                
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[selected_var], name='Avant', marker_color='#ef4444'))
                fig.add_trace(go.Box(y=df_clean[selected_var], name='Après', marker_color='#10b981'))
                fig.update_layout(title=f'Impact du Traitement des Outliers - {selected_var}')
                st.plotly_chart(fig, use_container_width=True)
    
    # === PAGE: ÉQUILIBRAGE ===
    elif page == "⚖️ Étape 6: Équilibrage des Données":
        st.markdown('<h2 class="section-header">⚖️ Équilibrage des Données</h2>', unsafe_allow_html=True)
        
        st.info("""
        **Problème**: Le dataset est fortement déséquilibré (4.87% de stroke).
        
        **Solutions disponibles**:
        - **Over-sampling**: SMOTE, ADASYN, Borderline-SMOTE, SVM-SMOTE, KMeans-SMOTE
        - **Under-sampling**: Random, Tomek Links, ENN, NearMiss
        - **Combinées**: SMOTE+Tomek, SMOTE+ENN
        """)
        
        df_processed, _ = preprocess_data(df)
        X = df_processed.drop('stroke', axis=1).values
        y = df_processed['stroke'].values
        
        tab1, tab2, tab3 = st.tabs(["🎯 Méthodes Over-sampling", "📉 Méthodes Under-sampling", "🔀 Méthodes Combinées"])
        
        with tab1:
            st.subheader("Over-sampling: Générer des Exemples Synthétiques")
            
            method = st.selectbox("Choisir une méthode:", 
                                ["SMOTE", "ADASYN", "Borderline-SMOTE", "SVM-SMOTE", "KMeans-SMOTE"])
            
            if st.button("Appliquer Over-sampling"):
                balancer = DataBalancer()
                
                method_map = {
                    "SMOTE": 'smote',
                    "ADASYN": 'adasyn',
                    "Borderline-SMOTE": 'borderline',
                    "SVM-SMOTE": 'svm-smote',
                    "KMeans-SMOTE": 'kmeans-smote'
                }
                
                X_balanced, y_balanced = balancer.balance_data(X, y, method=method_map[method])
                report = balancer.get_balancing_report()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Distribution Originale:**")
                    fig = create_class_distribution(y, "Distribution Originale")
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric("Classe 0", report['original'][0])
                    st.metric("Classe 1", report['original'][1])
                    st.metric("Ratio", f"1:{report['original_ratio']:.1f}")
                
                with col2:
                    st.write("**Distribution Équilibrée:**")
                    fig = create_class_distribution(y_balanced, f"Après {method}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric("Classe 0", report['balanced'][0])
                    st.metric("Classe 1", report['balanced'][1])
                    st.metric("Ratio", f"1:{report['balanced_ratio']:.1f}")
                
                st.success(f"**{method} appliqué avec succès!** Nouveau dataset: {len(X_balanced)} exemples.")
        
        with tab2:
            st.subheader("Under-sampling: Réduire la Classe Majoritaire")
            
            method = st.selectbox("Choisir une méthode:", 
                                ["Random Under-sampling", "Tomek Links", "ENN", "NearMiss"])
            
            if st.button("Appliquer Under-sampling"):
                balancer = DataBalancer()
                
                method_map = {
                    "Random Under-sampling": 'random-under',
                    "Tomek Links": 'tomek',
                    "ENN": 'enn',
                    "NearMiss": 'nearmiss'
                }
                
                X_balanced, y_balanced = balancer.balance_data(X, y, method=method_map[method])
                report = balancer.get_balancing_report()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = create_class_distribution(y, "Distribution Originale")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = create_class_distribution(y_balanced, f"Après {method}")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Dataset réduit de {len(X)} à {len(X_balanced)} exemples.")
        
        with tab3:
            st.subheader("Méthodes Combinées: Over + Under-sampling")
            
            method = st.selectbox("Choisir une méthode:", ["SMOTE+Tomek", "SMOTE+ENN"])
            
            if st.button("Appliquer Méthode Combinée"):
                balancer = DataBalancer()
                
                method_map = {
                    "SMOTE+Tomek": 'smote-tomek',
                    "SMOTE+ENN": 'smote-enn'
                }
                
                X_balanced, y_balanced = balancer.balance_data(X, y, method=method_map[method])
                report = balancer.get_balancing_report()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = create_class_distribution(y, "Distribution Originale")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = create_class_distribution(y_balanced, f"Après {method}")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"**{method}**: Génération synthétique + Nettoyage. Dataset final: {len(X_balanced)} exemples.")
    
    # === PAGE: MODÈLES ML ===
    elif page == "🤖 Étape 7: Modèles ML":
        st.markdown('<h2 class="section-header">🤖 Modèles de Machine Learning</h2>', unsafe_allow_html=True)
        
        # Préparation des données
        df_processed, _ = preprocess_data(df)
        X = df_processed.drop('stroke', axis=1)
        y = df_processed['stroke']
        
        # Option d'équilibrage
        use_balancing = st.checkbox("Utiliser SMOTE pour équilibrer les données", value=True)
        
        if use_balancing:
            balancer = DataBalancer()
            X_balanced, y_balanced = balancer.balance_data(X.values, y.values, method='smote')
            X = pd.DataFrame(X_balanced, columns=X.columns)
            y = pd.Series(y_balanced)
            st.success(f"Dataset équilibré: {len(X)} exemples")
        
        # Entraînement
        if st.button("Entraîner les Modèles"):
            with st.spinner("Entraînement en cours..."):
                trainer = ModelTrainer()
                X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
                
                results = trainer.train_all_models(X_train, y_train, X_test, y_test)
                
                st.success("Entraînement terminé!")
                
                # Résultats
                st.subheader("📊 Résultats des Modèles")
                results_df = trainer.get_results_dataframe()
                st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                
                # Visualisations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(results_df.reset_index(), x='index', y='f1_score',
                               title='Comparaison F1-Score', labels={'index': 'Modèle'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(results_df.reset_index(), x='index', y='roc_auc',
                               title='Comparaison ROC-AUC', labels={'index': 'Modèle'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Meilleur modèle
                best_model_name = results_df['f1_score'].idxmax()
                st.success(f"🏆 **Meilleur Modèle**: {best_model_name} (F1={results_df.loc[best_model_name, 'f1_score']:.4f})")
                
                # Feature Importance
                st.subheader("🔍 Importance des Variables")
                importances = trainer.get_feature_importance('Random Forest')
                if importances is not None:
                    fig = create_feature_importance(importances, X.columns)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Matrice de confusion
                st.subheader("📊 Matrice de Confusion - Meilleur Modèle")
                y_pred = results[best_model_name]['y_pred']
                fig = create_confusion_matrix(y_test, y_pred)
                st.plotly_chart(fig, use_container_width=True)
                
                # Courbe ROC
                st.subheader("📈 Courbe ROC")
                y_pred_proba = results[best_model_name]['y_pred_proba']
                if y_pred_proba is not None:
                    fig = create_roc_curve(y_test, y_pred_proba, best_model_name)
                    st.plotly_chart(fig, use_container_width=True)
    
    # === PAGE: PRÉDICTION ===
    elif page == "🔮 Prédiction en Temps Réel":
        st.markdown('<h2 class="section-header">🔮 Prédiction en Temps Réel</h2>', unsafe_allow_html=True)
        
        st.info("Entraînez d'abord les modèles dans l'étape précédente, puis utilisez cet outil pour faire des prédictions.")
        
        st.write("⚠️ Fonctionnalité de prédiction en temps réel à implémenter avec sauvegarde du modèle.")

if __name__ == "__main__":
    main()
