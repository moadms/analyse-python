@echo off
echo 🏥 Démarrage de l'Application d'Analyse Stroke ML...
echo.

cd /d "%~dp0stroke-ml-app"

echo 📁 Dossier courant: %cd%
echo.

echo 🔧 Activation environnement virtuel...
call "%~dp0.venv\Scripts\activate.bat"

echo.
echo 🚀 Lancement de Streamlit...
echo 🌐 L'application s'ouvrira dans votre navigateur
echo 📊 URL: http://localhost:8501
echo.
echo ❌ Pour arrêter l'application, fermez cette fenêtre ou appuyez Ctrl+C
echo.

streamlit run app.py --server.port 8501

pause