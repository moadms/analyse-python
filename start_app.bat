@echo off
chcp 65001 > nul
echo 🏥 Démarrage de l'Application d'Analyse Stroke ML...
echo.

cd /d "%~dp0"

echo 📁 Dossier courant: %cd%
echo.

if exist "%~dp0.venv\Scripts\activate.bat" (
    echo 🔧 Activation environnement virtuel...
    call "%~dp0.venv\Scripts\activate.bat"
) else (
    echo ⚠️ Environnement virtuel introuvable dans .venv, utilisation de Python global...
)

echo.
echo 🚀 Lancement de Streamlit...
echo 🌐 L'application s'ouvre dans votre navigateur :
echo 📊 URL: http://localhost:8501
echo.
echo ❌ Pour arrêter l'application, fermez cette fenêtre ou appuyez sur Ctrl+C
echo.

streamlit run app.py --server.port 8501

pause
