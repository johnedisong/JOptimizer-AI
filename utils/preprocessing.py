#Módulo para preprocesamiento de datos.

import pandas as pd
import numpy as np
from typing import Dict, Tuple

def analyze_code_metrics(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Analiza las métricas de código y proporciona un resumen estadístico.

    Args:
        file_path (str): Ruta al archivo CSV con las métricas.

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame con los datos y diccionario con estadísticas.
    """
    # Leer los datos
    data = pd.read_csv(file_path)

    # Estadísticas básicas
    stats = {
        'total_samples': len(data),
        'optimal_code': sum(data['is_optimal']),
        'suboptimal_code': len(data) - sum(data['is_optimal']),
        'metrics_summary': data.describe().to_dict(),
        'correlations': data.corr()['is_optimal'].to_dict()
    }

    # Calcular umbrales de las métricas
    thresholds = {}
    for column in data.columns:
        if column != 'is_optimal':
            optimal_values = data[data['is_optimal'] == 1][column]
            suboptimal_values = data[data['is_optimal'] == 0][column]

            thresholds[column] = {
                'optimal_mean': optimal_values.mean(),
                'optimal_std': optimal_values.std(),
                'suboptimal_mean': suboptimal_values.mean(),
                'suboptimal_std': suboptimal_values.std()
            }

    stats['metric_thresholds'] = thresholds

    return data, stats

if __name__ == "__main__":
    # Ejemplo de uso
    data, stats = analyze_code_metrics('../data/train/code_metrics.csv')

    print("\n=== Resumen de Métricas de Código ===")
    print(f"Total de muestras: {stats['total_samples']}")
    print(f"Código óptimo: {stats['optimal_code']} ({stats['optimal_code']/stats['total_samples']*100:.2f}%)")
    print(f"Código subóptimo: {stats['suboptimal_code']} ({stats['suboptimal_code']/stats['total_samples']*100:.2f}%)")

    print("\n=== Correlación con Optimización ===")
    correlations = stats['correlations']
    for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        if metric != 'is_optimal':
            print(f"{metric}: {corr:.3f}")

    print("\n=== Umbrales por Métrica ===")
    for metric, values in stats['metric_thresholds'].items():
        print(f"\n{metric}:")
        print(f"  Óptimo: {values['optimal_mean']:.2f} ± {values['optimal_std']:.2f}")
        print(f"  Subóptimo: {values['suboptimal_mean']:.2f} ± {values['suboptimal_std']:.2f}")