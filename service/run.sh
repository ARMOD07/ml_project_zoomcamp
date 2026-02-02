#!/bin/bash
# Fichier: start.sh
# Script de démarrage pour Railway

echo "=================================="
echo "   ML Prediction Service v1.0"
echo "=================================="

# Vérifier les dépendances
echo "🔍 Vérification des dépendances..."
pip list | grep -E "(fastapi|uvicorn|tensorflow|scikit)"

# Vérifier les fichiers de modèle
echo "📁 Fichiers disponibles:"
ls -la *.json *.h5 *.pkl 2>/dev/null || echo "Aucun fichier de modèle trouvé"

# Démarrer l'application
echo "🚀 Démarrage de l'application..."
echo "Port: ${PORT:-8080}"
python app.py