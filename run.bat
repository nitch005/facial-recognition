@echo off
REM ============================================================
REM  Lanceur Windows de l'application de reconnaissance faciale
REM  Cree l'environnement, installe les dependances et le modele
REM  UNE SEULE FOIS, puis lance l'application.
REM ============================================================
cd /d "%~dp0"

if not exist venv (
    echo ^>^> Creation de l'environnement virtuel...
    python -m venv venv
    echo ^>^> Installation des dependances...
    venv\Scripts\python -m pip install --upgrade pip
    venv\Scripts\pip install -r requirements.txt
)

echo ^>^> Verification / telechargement unique du modele...
venv\Scripts\python scripts\setup_model.py

echo ^>^> Lancement de l'application...
venv\Scripts\python main.py

pause
