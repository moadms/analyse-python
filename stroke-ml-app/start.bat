@echo off
chcp 65001 >nul
echo =========================================
echo   🏥 Application Prédiction AVC
echo   👨‍💻 Créé par MOAD MOUSSAOUI
echo =========================================
echo.

cd /d "%~dp0"

REM Vérifier si l'environnement virtuel existe
if exist "venv\Scripts\activate.bat" (
    echo [1/4] ✅ Activation environnement virtuel...
    call venv\Scripts\activate.bat
) else if exist "..\..\.venv\Scripts\activate.bat" (
    echo [1/4] ✅ Activation environnement virtuel (parent)...
    call ..\..\.venv\Scripts\activate.bat
) else (
    echo [1/4] ⚠️  Pas d'environnement virtuel trouvé
    echo        Utilisation Python système...
)

echo.
echo [2/4] 🔍 Vérification des packages...
python -c "import streamlit; import pandas; import sklearn; import prince; import xgboost; import imblearn; print('✓ Tous les packages sont installés!')" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo ⚠️  Certains packages manquent. Installation en cours...
    pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib imbalanced-learn prince scipy xgboost --quiet
    if %errorlevel% neq 0 (
        echo ❌ Erreur installation! Installez manuellement: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo ✅ Installation terminée!
)

echo.
echo [3/4] 📊 Vérification du dataset...
if exist "healthcare-dataset-stroke-data.csv" (
    echo ✅ Dataset trouvé: healthcare-dataset-stroke-data.csv
) else (
    echo ⚠️  Dataset non trouvé - sera téléchargé automatiquement au démarrage
)

echo.
echo [4/4] 🚀 Lancement de l'application Streamlit...
echo.
echo ┌──────────────────────────────────────────────┐
echo │  L'application va s'ouvrir dans le navigateur│
echo │  URL: http://localhost:8501                  │
echo │                                              │
echo │  Appuyez sur Ctrl+C pour arrêter            │
echo └──────────────────────────────────────────────┘
echo.

echo.
echo [3/3] Lancement de l'application...
echo.
echo ===================================
echo   Application disponible sur:
echo   http://localhost:8502
echo ===================================
echo.

streamlit run app.py --server.port 8502

pause
