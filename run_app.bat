@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Criando ambiente virtual...
    python -m venv .venv
)

echo Ativando ambiente virtual...
call .venv\Scripts\activate

echo Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Iniciando aplicativo...
python main.py
