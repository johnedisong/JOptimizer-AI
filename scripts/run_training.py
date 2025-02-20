#Script para ejecutar el entrenamiento y guardar modelo

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train_model import ModelTrainer
from core.model_handler import ModelHandler
import pandas as pd
import argparse

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_directories_exist():
    project_root = get_project_root()
    required_dirs = ['data/train', 'models']
    for directory in required_dirs:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Verificado directorio: {dir_path}")

def main():
    # Asegurar que existan los directorios necesarios
    ensure_directories_exist()
    project_root = get_project_root()

    parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación de código')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo CSV de entrenamiento')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'decision_tree'],
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--output', type=str, required=True, help='Ruta donde guardar el modelo')

    args = parser.parse_args()

    print(f"Entrenando modelo {args.model_type}...")

    try:
        # Convertir rutas relativas a absolutas
        data_path = os.path.join(project_root, args.data)
        output_path = os.path.join(project_root, args.output)

        # Cargar datos
        print(f"Cargando datos desde {data_path}")
        training_data = pd.read_csv(data_path)
        print(f"Datos cargados: {len(training_data)} muestras")
        print(f"Columnas disponibles: {training_data.columns.tolist()}")

        # Verificar que los datos son correctos
        if not isinstance(training_data, pd.DataFrame):
            raise TypeError("Error: Los datos no son un DataFrame")

        if 'is_optimal' not in training_data.columns:
            raise ValueError("Error: No se encontró la columna 'is_optimal' en los datos")

        # Crear entrenador y manejador de modelos
        trainer = ModelTrainer()
        handler = ModelHandler()

        # Entrenar modelo
        print("Iniciando entrenamiento...")
        model, metrics = trainer.train_model(training_data, args.model_type)

        # Asegurar que el directorio de salida existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Guardar modelo con métricas como metadata
        print(f"Guardando modelo en {output_path}...")
        handler.save_model(model, output_path, metrics)

        print(f"\nModelo entrenado y guardado exitosamente")
        print(f"Precisión en pruebas: {metrics['test_score']:.3f}")

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        print(f"Ruta del proyecto: {project_root}")
        print(f"Ruta de datos: {data_path}")
        print(f"Ruta de salida: {output_path}")
        raise

if __name__ == "__main__":
    main()