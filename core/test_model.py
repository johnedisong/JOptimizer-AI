#Módulo para prueba de modelos.

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import classification_report, confusion_matrix

class ModelTester:
    """
    Clase para probar modelos entrenados con nuevos datos.
    """

    def __init__(self):
        """Inicializa el ModelTester."""
        pass

    def test_model(self,
                   model: Any,
                   test_data: pd.DataFrame,
                   true_labels: pd.Series = None) -> Dict:
        """
        Prueba un modelo con nuevos datos.

        Args:
            model: Modelo entrenado
            test_data: DataFrame con las métricas de código a evaluar
            true_labels: Etiquetas reales (opcional, para evaluación)

        Returns:
            Dict: Resultados de las predicciones y métricas si hay etiquetas
        """
        if not isinstance(test_data, pd.DataFrame):
            raise TypeError("Los datos de prueba deben ser un DataFrame de pandas")

        # Verificar que tenemos las columnas necesarias
        expected_columns = [col for col in test_data.columns if col != 'is_optimal']
        if len(expected_columns) == 0:
            raise ValueError("El DataFrame de prueba no contiene las métricas necesarias")

        # Hacer predicciones
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)

        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'num_samples': len(test_data)
        }

        # Si tenemos etiquetas reales, calcular métricas
        if true_labels is not None:
            results.update({
                'accuracy': model.score(test_data, true_labels),
                'classification_report': classification_report(
                    true_labels,
                    predictions,
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(
                    true_labels,
                    predictions
                )
            })

        return results

    def analyze_predictions(self,
                          test_data: pd.DataFrame,
                          predictions: np.ndarray,
                          probabilities: np.ndarray) -> Dict:
        """
        Analiza las predicciones en detalle.

        Args:
            test_data: Datos de prueba
            predictions: Predicciones del modelo
            probabilities: Probabilidades de las predicciones

        Returns:
            Dict: Análisis detallado de las predicciones
        """
        # Combinar datos con predicciones
        results_df = test_data.copy()
        results_df['predicted_label'] = predictions
        results_df['confidence'] = np.max(probabilities, axis=1)

        # Análisis de confianza
        confidence_analysis = {
            'mean_confidence': results_df['confidence'].mean(),
            'min_confidence': results_df['confidence'].min(),
            'max_confidence': results_df['confidence'].max(),
            'std_confidence': results_df['confidence'].std()
        }

        # Contar predicciones por clase
        prediction_counts = {
            'optimal_count': sum(predictions == 1),
            'suboptimal_count': sum(predictions == 0)
        }

        # Identificar casos de baja confianza
        low_confidence_threshold = 0.7
        low_confidence_cases = results_df[
            results_df['confidence'] < low_confidence_threshold
        ]

        # Análisis por característica
        feature_analysis = {}
        for feature in test_data.columns:
            if feature != 'is_optimal':
                optimal_values = results_df[results_df['predicted_label'] == 1][feature]
                suboptimal_values = results_df[results_df['predicted_label'] == 0][feature]

                feature_analysis[feature] = {
                    'optimal_mean': optimal_values.mean(),
                    'optimal_std': optimal_values.std(),
                    'suboptimal_mean': suboptimal_values.mean(),
                    'suboptimal_std': suboptimal_values.std()
                }

        return {
            'confidence_analysis': confidence_analysis,
            'prediction_counts': prediction_counts,
            'low_confidence_cases': low_confidence_cases,
            'feature_analysis': feature_analysis,
            'full_results': results_df
        }

if __name__ == "__main__":
    # Ejemplo de uso
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    # Cargar datos de ejemplo
    print("Cargando datos de prueba...")
    try:
        test_data = pd.read_csv('../data/test/code_metrics.csv')

        # Separar etiquetas si existen
        if 'is_optimal' in test_data.columns:
            y_test = test_data['is_optimal']
            X_test = test_data.drop('is_optimal', axis=1)
        else:
            y_test = None
            X_test = test_data

        # Crear y entrenar un modelo dummy
        print("\nCreando modelo de prueba...")
        model = RandomForestClassifier(random_state=42)
        if y_test is not None:
            model.fit(X_test, y_test)

        # Crear tester y probar modelo
        print("\nProbando modelo...")
        tester = ModelTester()
        results = tester.test_model(model, X_test, y_test)

        # Analizar predicciones
        analysis = tester.analyze_predictions(
            X_test,
            results['predictions'],
            results['probabilities']
        )

        # Mostrar resultados
        print("\n=== Resultados de las Pruebas ===")
        if y_test is not None:
            print(f"Precisión: {results['accuracy']:.3f}")
            print("\nMatriz de Confusión:")
            print(results['confusion_matrix'])

        print("\nAnálisis de Confianza:")
        conf = analysis['confidence_analysis']
        print(f"Confianza media: {conf['mean_confidence']:.3f}")
        print(f"Confianza mínima: {conf['min_confidence']:.3f}")
        print(f"Confianza máxima: {conf['max_confidence']:.3f}")

        print("\nDistribución de Predicciones:")
        counts = analysis['prediction_counts']
        print(f"Códigos Óptimos: {counts['optimal_count']}")
        print(f"Códigos Subóptimos: {counts['suboptimal_count']}")

    except Exception as e:
        print(f"Error durante la prueba: {str(e)}")