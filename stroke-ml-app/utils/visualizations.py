"""
Module de visualisations pour l'analyse de stroke.
"""

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def create_distribution_plot(df, column, title=None):
    """Crée un histogramme avec courbe KDE."""
    fig = px.histogram(df, x=column, marginal='box', 
                      title=title or f'Distribution de {column}',
                      color_discrete_sequence=['#3b82f6'])
    return fig


def create_qq_plot(df, column):
    """Crée un Q-Q plot pour vérifier la normalité."""
    fig = go.Figure()
    
    theoretical_quantiles, ordered_values = stats.probplot(df[column].dropna(), dist="norm")
    
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles[0],
        y=theoretical_quantiles[1],
        mode='markers',
        name='Données',
        marker=dict(color='#3b82f6', size=6)
    ))
    
    # Ligne théorique
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles[0],
        y=theoretical_quantiles[0] * theoretical_quantiles[1][1] + theoretical_quantiles[1][0],
        mode='lines',
        name='Distribution normale',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Q-Q Plot - {column}',
        xaxis_title='Quantiles théoriques',
        yaxis_title='Quantiles observés'
    )
    
    return fig


def create_boxplot(df, column, by=None):
    """Crée un boxplot."""
    if by:
        fig = px.box(df, x=by, y=column, color=by,
                    title=f'Boxplot de {column} par {by}')
    else:
        fig = px.box(df, y=column, title=f'Boxplot de {column}')
    
    return fig


def create_correlation_heatmap(df, method='pearson'):
    """Crée une heatmap de corrélation."""
    corr = df.corr(method=method)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Corrélation")
    ))
    
    fig.update_layout(
        title=f'Matrice de Corrélation ({method.capitalize()})',
        width=800,
        height=800
    )
    
    return fig


def create_violin_plot(df, x, y, color=None):
    """Crée un violin plot."""
    fig = px.violin(df, x=x, y=y, color=color, box=True, points='outliers',
                   title=f'Distribution de {y} par {x}')
    return fig


def create_scatter_with_regression(df, x, y, color=None):
    """Crée un scatter plot avec ligne de régression."""
    fig = px.scatter(df, x=x, y=y, color=color, trendline='ols',
                    title=f'{y} vs {x}')
    return fig


def create_pairplot_plotly(df, columns, color=None):
    """Crée un pairplot interactif avec Plotly."""
    fig = px.scatter_matrix(df[columns], dimensions=columns, color=color,
                           title='Pairplot Interactif')
    fig.update_traces(diagonal_visible=False)
    return fig


def create_class_distribution(y, title='Distribution des Classes'):
    """Visualise la distribution des classes."""
    class_counts = pd.Series(y).value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(x=class_counts.index.astype(str), y=class_counts.values,
               text=class_counts.values, textposition='auto',
               marker_color=['#4ade80', '#ef4444'])
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Classe',
        yaxis_title='Nombre d\'exemples',
        showlegend=False
    )
    
    return fig


def create_transformation_comparison(df_original, df_transformed, column, transformation_name):
    """Compare une variable avant et après transformation."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_original[column],
        name='Original',
        opacity=0.7,
        marker_color='#3b82f6'
    ))
    
    transformed_col = f'{column}_{transformation_name}'
    if transformed_col in df_transformed.columns:
        fig.add_trace(go.Histogram(
            x=df_transformed[transformed_col],
            name=f'Transformé ({transformation_name})',
            opacity=0.7,
            marker_color='#10b981'
        ))
    
    fig.update_layout(
        title=f'Comparaison: {column} (Original vs {transformation_name})',
        xaxis_title='Valeur',
        yaxis_title='Fréquence',
        barmode='overlay'
    )
    
    return fig


def create_skewness_comparison(skewness_before, skewness_after):
    """Compare le skewness avant et après transformation."""
    df_comparison = pd.DataFrame({
        'Variable': skewness_before.index,
        'Avant': skewness_before.values.flatten(),
        'Après': skewness_after.values.flatten()
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_comparison['Variable'],
        y=df_comparison['Avant'],
        name='Avant transformation',
        marker_color='#ef4444'
    ))
    
    fig.add_trace(go.Bar(
        x=df_comparison['Variable'],
        y=df_comparison['Après'],
        name='Après transformation',
        marker_color='#10b981'
    ))
    
    fig.update_layout(
        title='Comparaison du Skewness (Asymétrie)',
        xaxis_title='Variable',
        yaxis_title='Skewness',
        barmode='group'
    )
    
    return fig


def create_outlier_visualization(df, column, outliers_indices):
    """Visualise les outliers détectés."""
    fig = go.Figure()
    
    # Points normaux
    normal_indices = df.index.difference(outliers_indices)
    fig.add_trace(go.Scatter(
        x=normal_indices,
        y=df.loc[normal_indices, column],
        mode='markers',
        name='Valeurs normales',
        marker=dict(color='#3b82f6', size=5)
    ))
    
    # Outliers
    if len(outliers_indices) > 0:
        fig.add_trace(go.Scatter(
            x=outliers_indices,
            y=df.loc[outliers_indices, column],
            mode='markers',
            name='Outliers',
            marker=dict(color='#ef4444', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title=f'Détection des Outliers - {column}',
        xaxis_title='Index',
        yaxis_title=column
    )
    
    return fig


def create_roc_curve(y_true, y_pred_proba, model_name='Model'):
    """Crée une courbe ROC."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        mode='lines',
        line=dict(color='#3b82f6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Baseline (AUC = 0.500)',
        mode='lines',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Courbe ROC - {model_name}',
        xaxis_title='Taux de Faux Positifs',
        yaxis_title='Taux de Vrais Positifs',
        width=600,
        height=600
    )
    
    return fig


def create_confusion_matrix(y_true, y_pred, labels=['No Stroke', 'Stroke']):
    """Crée une matrice de confusion."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis_title='Prédit',
        yaxis_title='Réel',
        width=500,
        height=500
    )
    
    return fig


def create_feature_importance(importances, feature_names, top_n=10):
    """Visualise l'importance des features."""
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig = px.bar(df_importance, y='Feature', x='Importance',
                orientation='h',
                title=f'Top {top_n} Features Importantes',
                color='Importance',
                color_continuous_scale='Viridis')
    
    return fig
