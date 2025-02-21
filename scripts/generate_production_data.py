# scripts/generate_production_data.py

import pandas as pd
import numpy as np

def generate_production_metrics(num_samples=10):
    """
    Genera métricas que simulan código Java real para pruebas en producción.
    """
    # Simulando diferentes tipos de clases Java
    class_types = ['Controller', 'Service', 'Repository', 'Model', 'Util']
    metrics = []

    for _ in range(num_samples):
        class_type = np.random.choice(class_types)

        # Ajustar métricas según el tipo de clase
        if class_type == 'Controller':
            # Controllers tienden a tener más métodos pero menos complejidad
            metrics.append({
                'lines_of_code': np.random.randint(100, 300),
                'effective_lines': np.random.randint(80, 250),
                'number_of_methods': np.random.randint(5, 15),
                'cyclomatic_complexity': np.random.randint(1, 10),
                'inheritance_depth': np.random.randint(1, 3),
                'number_of_branches': np.random.randint(10, 30),
                'coupling_between_objects': np.random.randint(3, 8),
                'external_dependencies': np.random.randint(2, 6),
                'lack_of_cohesion': np.random.uniform(0.2, 0.5),
                'class_type': class_type
            })
        elif class_type == 'Service':
            # Services pueden ser complejos
            metrics.append({
                'lines_of_code': np.random.randint(200, 500),
                'effective_lines': np.random.randint(150, 400),
                'number_of_methods': np.random.randint(10, 20),
                'cyclomatic_complexity': np.random.randint(5, 25),
                'inheritance_depth': np.random.randint(1, 4),
                'number_of_branches': np.random.randint(20, 50),
                'coupling_between_objects': np.random.randint(5, 12),
                'external_dependencies': np.random.randint(3, 8),
                'lack_of_cohesion': np.random.uniform(0.3, 0.7),
                'class_type': class_type
            })
        elif class_type == 'Repository':
            # Repositories suelen ser simples
            metrics.append({
                'lines_of_code': np.random.randint(50, 150),
                'effective_lines': np.random.randint(40, 120),
                'number_of_methods': np.random.randint(3, 8),
                'cyclomatic_complexity': np.random.randint(1, 5),
                'inheritance_depth': np.random.randint(1, 2),
                'number_of_branches': np.random.randint(5, 15),
                'coupling_between_objects': np.random.randint(2, 5),
                'external_dependencies': np.random.randint(1, 4),
                'lack_of_cohesion': np.random.uniform(0.1, 0.4),
                'class_type': class_type
            })
        elif class_type == 'Model':
            # Models son mayormente datos
            metrics.append({
                'lines_of_code': np.random.randint(30, 200),
                'effective_lines': np.random.randint(25, 150),
                'number_of_methods': np.random.randint(2, 10),
                'cyclomatic_complexity': np.random.randint(1, 3),
                'inheritance_depth': np.random.randint(0, 2),
                'number_of_branches': np.random.randint(0, 5),
                'coupling_between_objects': np.random.randint(1, 4),
                'external_dependencies': np.random.randint(0, 3),
                'lack_of_cohesion': np.random.uniform(0.1, 0.3),
                'class_type': class_type
            })
        else:  # Util
            # Utility classes varían
            metrics.append({
                'lines_of_code': np.random.randint(100, 400),
                'effective_lines': np.random.randint(80, 300),
                'number_of_methods': np.random.randint(5, 15),
                'cyclomatic_complexity': np.random.randint(3, 15),
                'inheritance_depth': np.random.randint(0, 2),
                'number_of_branches': np.random.randint(10, 40),
                'coupling_between_objects': np.random.randint(2, 7),
                'external_dependencies': np.random.randint(1, 5),
                'lack_of_cohesion': np.random.uniform(0.2, 0.6),
                'class_type': class_type
            })

    # Convertir a DataFrame
    df = pd.DataFrame(metrics)

    # Asegurar que effective_lines <= lines_of_code
    df['effective_lines'] = df.apply(
        lambda row: min(row['effective_lines'], row['lines_of_code']),
        axis=1
    )

    return df

if __name__ == "__main__":
    # Generar datos de producción
    print("Generando datos de producción...")
    production_data = generate_production_metrics(10)

    # Guardar datos
    output_file = 'data/test/production_metrics.csv'
    production_data.to_csv(output_file, index=False)

    print("\nEstadísticas de los datos generados:")
    print("\nDistribución por tipo de clase:")
    print(production_data['class_type'].value_counts())

    print("\nEstadísticas descriptivas:")
    print(production_data.describe())

    print(f"\nDatos guardados en: {output_file}")