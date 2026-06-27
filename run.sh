#!/usr/bin/env bash
#
# Lance l'application de reconnaissance faciale.
# Cree l'environnement virtuel et installe les dependances au premier lancement.
#
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo ">> Creation de l'environnement virtuel..."
  python3 -m venv venv
  echo ">> Installation des dependances..."
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install -r requirements.txt
fi

echo ">> Lancement de l'application..."
./venv/bin/python main.py
