import os
import json

def create_directory(path):
    """Crea un directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creado directorio: {path}")

def create_file(path, content=""):
    """Crea un archivo con contenido opcional."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Creado archivo: {path}")

def setup_project():
    # Directorios principales
    directories = [
        'core',
        'utils',
        'data/train',
        'data/test',
        'models',
        'scripts',
        'config',
        'tests'
    ]

    # Crear directorios
    for directory in directories:
        create_directory(directory)

    # Crear archivos __init__.py
    init_files = [
        'core/__init__.py',
        'utils/__init__.py',
        'tests/__init__.py'
    ]
    for init_file in init_files:
        create_file(init_file)

    # Crear archivos principales
    core_files = {
        'core/train_model.py': '"""Módulo para entrenamiento de modelos."""\n',
        'core/test_model.py': '"""Módulo para prueba de modelos."""\n',
        'core/model_handler.py': '"""Módulo para gestión de modelos."""\n'
    }
    for file_path, content in core_files.items():
        create_file(file_path, content)

    # Crear archivo de utilidades
    create_file('utils/preprocessing.py', '"""Módulo para preprocesamiento de datos."""\n')

    # Crear scripts
    scripts = {
        'scripts/run_training.py': '"""Script para ejecutar el entrenamiento."""\n',
        'scripts/run_testing.py': '"""Script para ejecutar las pruebas."""\n'
    }
    for file_path, content in scripts.items():
        create_file(file_path, content)

    # Crear archivo de configuración
    config = {
        "models": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "decision_tree": {
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42
        },
        "preprocessing": {
            "normalize": True,
            "remove_outliers": True
        }
    }
    create_file('config/parameters.json', json.dumps(config, indent=4))

    # Crear README.md
    readme_content = """# JOptimizer-AI

Sistema de análisis de código Java que utiliza machine learning para clasificar e identificar oportunidades de optimización. Implementa Random Forest y Árboles de Decisión para determinar si los segmentos de código son óptimos o subóptimos, basándose en métricas de código y datos históricos.

## Estructura del Proyecto
- `core/`: Módulos principales del sistema
- `utils/`: Utilidades y funciones auxiliares
- `data/`: Datos para entrenamiento y pruebas
- `models/`: Modelos entrenados
- `scripts/`: Scripts ejecutables
- `config/`: Archivos de configuración
- `tests/`: Pruebas unitarias y de integración

## Desarrollado por:
- Juan Sebastian Fernando Veloza Ceron
- Angelica Daniela Quevedo Cortes
- Alvaro Urrego Viana
- John Edison Goyeneche Barbosa
"""
    create_file('README.md', readme_content)

    # Crear requirements.txt
    requirements = """numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
joblib>=1.1.0
pytest>=6.2.0"""
    create_file('requirements.txt', requirements)

if __name__ == "__main__":
    setup_project()