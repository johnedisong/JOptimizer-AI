# script que genera datos de forma balanceado para codigo optimo y suboptimo

import numpy as np
import pandas as pd

def generate_optimal_code_metrics(num_samples):
    """
    Genera métricas para código óptimo basado en buenas prácticas
    """
    return pd.DataFrame({
        'lines_of_code': np.random.normal(200, 50, num_samples).astype(int),
        'effective_lines': np.random.normal(150, 30, num_samples).astype(int),
        'number_of_methods': np.random.normal(5, 2, num_samples).astype(int),
        'cyclomatic_complexity': np.random.normal(8, 2, num_samples).astype(int),
        'inheritance_depth': np.random.normal(2, 1, num_samples).astype(int),
        'number_of_branches': np.random.normal(20, 5, num_samples).astype(int),
        'coupling_between_objects': np.random.normal(5, 2, num_samples).astype(int),
        'external_dependencies': np.random.normal(4, 1, num_samples).astype(int),
        'lack_of_cohesion': np.random.normal(0.25, 0.1, num_samples),
        'is_optimal': 1
    })

def generate_suboptimal_code_metrics(num_samples):
    """
    Genera métricas para código subóptimo con valores más problemáticos
    """
    return pd.DataFrame({
        'lines_of_code': np.random.normal(400, 100, num_samples).astype(int),
        'effective_lines': np.random.normal(300, 80, num_samples).astype(int),
        'number_of_methods': np.random.normal(15, 5, num_samples).astype(int),
        'cyclomatic_complexity': np.random.normal(20, 5, num_samples).astype(int),
        'inheritance_depth': np.random.normal(4, 2, num_samples).astype(int),
        'number_of_branches': np.random.normal(40, 10, num_samples).astype(int),
        'coupling_between_objects': np.random.normal(12, 3, num_samples).astype(int),
        'external_dependencies': np.random.normal(8, 2, num_samples).astype(int),
        'lack_of_cohesion': np.random.normal(0.6, 0.15, num_samples),
        'is_optimal': 0
    })

def clean_and_validate_data(df):
    """
    Limpia y valida los datos generados
    """
    # Asegurar que no hay valores negativos
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'is_optimal':
            df[col] = df[col].clip(lower=0)

    # Asegurar que lack_of_cohesion está entre 0 y 1
    df['lack_of_cohesion'] = df['lack_of_cohesion'].clip(lower=0, upper=1)

    # Asegurar que effective_lines <= lines_of_code
    df['effective_lines'] = df.apply(
        lambda row: min(row['effective_lines'], row['lines_of_code']),
        axis=1
    )

    return df

def generate_balanced_dataset(total_samples=1000):
    """
    Genera un conjunto de datos balanceado
    """
    # Generar igual número de muestras para cada clase
    samples_per_class = total_samples // 2

    # Generar datos
    optimal_data = generate_optimal_code_metrics(samples_per_class)
    suboptimal_data = generate_suboptimal_code_metrics(samples_per_class)

    # Combinar y mezclar datos
    all_data = pd.concat([optimal_data, suboptimal_data])
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Limpiar y validar
    all_data = clean_and_validate_data(all_data)

    return all_data

if __name__ == "__main__":
    # Generar dataset balanceado
    print("Generando conjunto de datos balanceado...")
    data = generate_balanced_dataset(1000)

    # Guardar datos
    data.to_csv('data/train/code_metrics.csv', index=False)

    # Mostrar estadísticas
    print("\nEstadísticas del conjunto de datos generado:")
    print(f"Total de muestras: {len(data)}")
    print("\nDistribución de clases:")
    class_dist = data['is_optimal'].value_counts()
    print(f"Óptimo (1): {class_dist[1]} muestras ({class_dist[1]/len(data)*100:.1f}%)")
    print(f"Subóptimo (0): {class_dist[0]} muestras ({class_dist[0]/len(data)*100:.1f}%)")

    print("\nEstadísticas descriptivas:")
    print(data.describe())

    print("\n¡Datos generados y guardados exitosamente!")