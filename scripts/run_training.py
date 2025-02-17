#Script para ejecutar el entrenamiento y guardar modelo

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train_model import ModelTrainer
from core.model_handler import ModelHandler
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación de código')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo CSV de entrenamiento')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'decision_tree'],
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--output', type=str, required=True, help='Ruta donde guardar el modelo')

    args = parser.parse_args()

    print(f"Entrenando modelo {args.model_type}...")

    try:
        # Cargar datos
        print(f"Cargando datos desde {args.data}")
        training_data = pd.read_csv(args.data)
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

        # Guardar modelo con métricas como metadata
        print("Guardando modelo...")
        handler.save_model(model, args.output, metrics)

        print(f"\nModelo entrenado y guardado en: {args.output}")
        print(f"Precisión en pruebas: {metrics['test_score']:.3f}")

        # Mostrar matriz de confusión
        print("\nMatriz de Confusión:")
        print(metrics['confusion_matrix'])

        if args.model_type == 'random_forest':
            print("\nImportancia de características:")
            for feature, importance in sorted(
                metrics['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"{feature}: {importance:.3f}")

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()