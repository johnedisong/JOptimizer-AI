#Script para ejecutar las pruebas.


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_handler import ModelHandler
from core.test_model import ModelTester
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Probar modelo de clasificación de código')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo guardado')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo CSV con datos para probar')

    args = parser.parse_args()

    print("Cargando modelo y datos...")

    # Cargar modelo
    handler = ModelHandler()
    model, metadata = handler.load_model(args.model)

    # Cargar datos
    data = pd.read_csv(args.data)

    # Crear tester y probar modelo
    tester = ModelTester()

    if 'is_optimal' in data.columns:
        X_test = data.drop('is_optimal', axis=1)
        y_test = data['is_optimal']
        results = tester.test_model(model, X_test, y_test)

        print("\nResultados de las pruebas:")
        print(f"Precisión: {results['accuracy']:.3f}")
        print("\nMatriz de Confusión:")
        print(results['confusion_matrix'])
    else:
        X_test = data
        results = tester.test_model(model, X_test)

        print("\nPredicciones:")
        for i, pred in enumerate(results['predictions']):
            prob = results['probabilities'][i][1]
            print(f"Muestra {i+1}: {'Óptimo' if pred == 1 else 'Subóptimo'} "
                  f"(Confianza: {prob:.2f})")

if __name__ == "__main__":
    main()