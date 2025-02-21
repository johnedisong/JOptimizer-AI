# App.py

from AIlibrary import AILibrary
import pandas as pd
import os

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

def ensure_directories():
    project_root = get_project_root()
    directories = ['models', 'data/train', 'data/test']

    created_dirs = {}
    for directory in directories:
        abs_dir = os.path.join(project_root, directory)
        os.makedirs(abs_dir, exist_ok=True)
        created_dirs[directory] = abs_dir
        print(f"Verificado directorio: {abs_dir}")

    return created_dirs

def main():
    """
    Ejemplo de uso de la librería AILibrary.
    """
    print("=== Ejemplo de Uso de JOptimizer-AI ===\n")

    # Obtener ruta raíz del proyecto
    project_root = get_project_root()
    print(f"Directorio raíz del proyecto: {project_root}")

    # Asegurar que existan los directorios necesarios
    directories = ensure_directories()

    # Construir rutas absolutas
    train_data_path = os.path.join(project_root, 'data', 'train', 'code_metrics.csv')
    test_data_path = os.path.join(project_root, 'data', 'test', 'code_metrics_test_final.csv')
    model_path = os.path.join(project_root, 'models', 'example_model.joblib')
    model_path_example = ""

    # Inicializar la librería
    ai_lib = AILibrary()

    try:
        # Verificar que exista el archivo de entrenamiento
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {train_data_path}")

        # 1. Entrenar un nuevo modelo
        print("\n1. Entrenando modelo...")
        print(f"Usando datos de: {train_data_path}")
        metrics = ai_lib.train(train_data_path, 'random_forest')
        print("\nModelo entrenado con éxito!")
        print(f"Precisión en pruebas: {metrics['test_score']:.3f}")

        # Guardar el modelo
        print(f"\nGuardando modelo en: {model_path}")
        model_path_example =  ai_lib.save_model(model_path)
        print("Modelo guardado exitosamente!   " + model_path_example)


        # 2. Cargar modelo existente
        print("\n2. Cargando modelo guardado...")
        ai_lib.load_model(model_path_example)
        print("Modelo cargado exitosamente!")

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"No se encontró el archivo de prueba: {test_data_path}")

        # 3. Analizar nuevo código
        print("\n3. Analizando nuevo código...")
        print(f"Usando datos de prueba de: {test_data_path}")
        test_data = pd.read_csv(test_data_path)

        features_to_use = ['lines_of_code', 'effective_lines', 'number_of_methods',
                        'cyclomatic_complexity', 'inheritance_depth', 'number_of_branches',
                        'coupling_between_objects', 'external_dependencies', 'lack_of_cohesion']
        missing_features = [f for f in features_to_use if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"Faltan las siguientes características en los datos de prueba: {missing_features}")
        test_data_features = test_data[features_to_use]
        results, analysis = ai_lib.analyze_code(test_data_features)


        # 4. Mostrar resultados
        print("\n=== Resultados del Análisis ===")
        predictions = results['predictions']
        probabilities = results['probabilities']

        print(f"\nTotal de clases analizadas: {len(predictions)}")
        print(f"Clases óptimas: {sum(predictions == 1)}")
        print(f"Clases subóptimas: {sum(predictions == 0)}")

        # 5. Obtener sugerencias de optimización
        print("\n=== Sugerencias de Optimización ===")
        suggestions = ai_lib.get_optimization_suggestions(test_data, results)

        for suggestion in suggestions:
            print(f"\nClase {suggestion['index'] + 1}:")
            print("Problemas detectados:")
            for issue in suggestion['issues']:
                print(f"- {issue}")
            print("Sugerencias:")
            for sug in suggestion['suggestions']:
                print(f"- {sug}")

        # 6. Mostrar métricas detalladas
        print("\n=== Métricas del Modelo ===")
        model_metrics = ai_lib.get_model_metrics()
        if 'feature_importance' in model_metrics:
            print("\nImportancia de características:")
            for feature, importance in sorted(
                model_metrics['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"{feature}: {importance:.3f}")

    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nVerifica que los siguientes archivos existan:")
        print(f"1. Datos de entrenamiento: {train_data_path}")
        print(f"2. Datos de prueba: {test_data_path}")
        print("\nEstructura de directorios actual:")
        for dir_name, dir_path in directories.items():
            print(f"- {dir_name}: {dir_path}")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
        print("Detalles de las rutas:")
        print(f"- Directorio raíz: {project_root}")
        print(f"- Archivo de entrenamiento: {train_data_path}")
        print(f"- Archivo de prueba: {test_data_path}")
        print(f"- Archivo de modelo: {model_path_example}")

if __name__ == "__main__":
    main()