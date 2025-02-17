# scripts/run_production_test.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_handler import ModelHandler
from core.test_model import ModelTester
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Probar modelo con datos de producción')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo guardado')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo CSV con datos para probar')

    args = parser.parse_args()

    print("=== Análisis de Código Java ===")
    print("\nCargando modelo y datos...")

    try:
        # Cargar modelo
        handler = ModelHandler()
        model, metadata = handler.load_model(args.model)

        # Cargar datos
        data = pd.read_csv(args.data)
        print(f"\nAnalizando {len(data)} clases Java...")

        # Guardar tipo de clase si existe
        class_types = data['class_type'] if 'class_type' in data.columns else None

        # Preparar datos para predicción
        if 'class_type' in data.columns:
            data = data.drop('class_type', axis=1)

        # Crear tester y probar modelo
        tester = ModelTester()
        results = tester.test_model(model, data)

        # Análisis detallado
        analysis = tester.analyze_predictions(
            data,
            results['predictions'],
            results['probabilities']
        )

        # Mostrar resultados
        print("\n=== Resultados del Análisis ===")

        for i in range(len(data)):
            prediction = "ÓPTIMO" if results['predictions'][i] == 1 else "SUBÓPTIMO"
            confidence = results['probabilities'][i][1] if results['predictions'][i] == 1 else results['probabilities'][i][0]

            print(f"\nClase {i+1}:")
            if class_types is not None:
                print(f"Tipo: {class_types.iloc[i]}")
            print(f"Clasificación: {prediction}")
            print(f"Confianza: {confidence:.2%}")

            # Si es subóptimo, mostrar métricas problemáticas
            if results['predictions'][i] == 0:
                print("Métricas problemáticas:")
                row = data.iloc[i]
                if row['cyclomatic_complexity'] > 15:
                    print(f"- Complejidad ciclomática alta: {row['cyclomatic_complexity']}")
                if row['coupling_between_objects'] > 8:
                    print(f"- Alto acoplamiento: {row['coupling_between_objects']}")
                if row['lack_of_cohesion'] > 0.5:
                    print(f"- Baja cohesión: {row['lack_of_cohesion']:.2f}")
                if row['number_of_methods'] > 15:
                    print(f"- Demasiados métodos: {row['number_of_methods']}")

        # Resumen general
        print("\n=== Resumen General ===")
        counts = analysis['prediction_counts']
        total = len(data)
        print(f"Total de clases analizadas: {total}")
        print(f"Clases óptimas: {counts['optimal_count']} ({counts['optimal_count']/total:.1%})")
        print(f"Clases subóptimas: {counts['suboptimal_count']} ({counts['suboptimal_count']/total:.1%})")

        conf = analysis['confidence_analysis']
        print(f"\nConfianza promedio: {conf['mean_confidence']:.1%}")

    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        raise

if __name__ == "__main__":
    main()