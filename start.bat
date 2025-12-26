@echo off
echo =========================================
echo   Application Stroke ML
echo =========================================
echo.

cd /d "%~dp0"

echo [1/3] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

echo.
echo [2/3] Verification des packages...
python -c "import streamlit; import pandas; import sklearn; import prince; print('✓ Tous les packages sont installes!')"

if %errorlevel% neq 0 (
    echo.
    echo ⚠ Certains packages manquent. Installation...
    pip install -r requirements.txt
)

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
