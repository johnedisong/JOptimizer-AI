# scripts/test_training.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train_model import ModelTrainer
import pandas as pd

def test_training():
    print("=== Prueba de Entrenamiento de Modelos ===")

    # 1. Cargar datos
    print("\nCargando datos...")
    try:
        data = pd.read_csv('../data/train/code_metrics.csv')
        print(f"Datos cargados exitosamente. Shape: {data.shape}")
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return

    # 2. Crear instancia del entrenador
    print("\nCreando entrenador...")
    trainer = ModelTrainer(random_state=42)

    # 3. Probar Random Forest
    print("\n--- Entrenando Random Forest ---")
    try:
        rf_model, rf_metrics = trainer.train_model(data, 'random_forest')

        print("\nMétricas de Random Forest:")
        print(f"Precisión en entrenamiento: {rf_metrics['train_score']:.3f}")
        print(f"Precisión en pruebas: {rf_metrics['test_score']:.3f}")

        print("\nImportancia de características:")
        for feature, importance in sorted(
            rf_metrics['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"{feature}: {importance:.3f}")

        print("\nMatriz de Confusión Random Forest:")
        print(rf_metrics['confusion_matrix'])

    except Exception as e:
        print(f"Error en Random Forest: {str(e)}")

    # 4. Probar Decision Tree
    print("\n--- Entrenando Decision Tree ---")
    try:
        dt_model, dt_metrics = trainer.train_model(data, 'decision_tree')

        print("\nMétricas de Decision Tree:")
        print(f"Precisión en entrenamiento: {dt_metrics['train_score']:.3f}")
        print(f"Precisión en pruebas: {dt_metrics['test_score']:.3f}")

        print("\nMatriz de Confusión Decision Tree:")
        print(dt_metrics['confusion_matrix'])

    except Exception as e:
        print(f"Error en Decision Tree: {str(e)}")

    # 5. Probar cambio de parámetros
    print("\n--- Probando cambio de parámetros en Random Forest ---")
    try:
        # Ver parámetros actuales
        current_params = trainer.get_model_params('random_forest')
        print("\nParámetros actuales RF:", current_params)

        # Cambiar algunos parámetros
        new_params = {
            'n_estimators': 200,
            'max_depth': 15
        }
        trainer.set_model_params('random_forest', new_params)

        # Entrenar con nuevos parámetros
        rf_model_new, rf_metrics_new = trainer.train_model(data, 'random_forest')

        print("\nMétricas con nuevos parámetros:")
        print(f"Precisión en entrenamiento: {rf_metrics_new['train_score']:.3f}")
        print(f"Precisión en pruebas: {rf_metrics_new['test_score']:.3f}")

    except Exception as e:
        print(f"Error al cambiar parámetros: {str(e)}")

if __name__ == "__main__":
    test_training()