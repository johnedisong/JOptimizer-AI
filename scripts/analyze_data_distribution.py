# scripts que saca metricas de la distribuación de la data de entrenamiento y testing

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def analyze_data_distribution():
    print("=== Análisis de Distribución de Datos ===\n")

    # Cargar datos
    try:
        data = pd.read_csv('../data/train/code_metrics.csv')
        print(f"Total de muestras: {len(data)}")

        # Análisis de clases
        class_distribution = data['is_optimal'].value_counts()
        print("\nDistribución de clases:")
        print(f"Subóptimo (0): {class_distribution.get(0, 0)} muestras ({class_distribution.get(0, 0)/len(data)*100:.2f}%)")
        print(f"Óptimo (1): {class_distribution.get(1, 0)} muestras ({class_distribution.get(1, 0)/len(data)*100:.2f}%)")

        if 0 in class_distribution and 1 in class_distribution:
            print(f"Ratio óptimo/subóptimo: {class_distribution[1]/class_distribution[0]:.2f}")

        # Estadísticas por clase
        print("\nEstadísticas por clase:")
        metrics = [col for col in data.columns if col != 'is_optimal']

        for metric in metrics:
            print(f"\n{metric}:")
            optimal = data[data['is_optimal'] == 1][metric]
            suboptimal = data[data['is_optimal'] == 0][metric]

            print("  Código Óptimo:")
            print(f"    Media: {optimal.mean():.2f}")
            print(f"    Mediana: {optimal.median():.2f}")
            print(f"    Desv. Est.: {optimal.std():.2f}")

            print("  Código Subóptimo:")
            print(f"    Media: {suboptimal.mean():.2f}")
            print(f"    Mediana: {suboptimal.median():.2f}")
            print(f"    Desv. Est.: {suboptimal.std():.2f}")

        # Análisis de correlaciones con is_optimal
        correlations = data.corr()['is_optimal'].sort_values(ascending=False)
        print("\nCorrelaciones con is_optimal:")
        for metric, corr in correlations.items():
            if metric != 'is_optimal':
                print(f"{metric}: {corr:.3f}")

    except Exception as e:
        print(f"Error al analizar datos: {str(e)}")

if __name__ == "__main__":
    analyze_data_distribution()