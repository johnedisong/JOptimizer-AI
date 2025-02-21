# AIlibrary.py Tiene la implementación principal de toda la libreria, metodos de entrenamiento analisis y Optimización
# AIlibrary.py

import os
from core.train_model import ModelTrainer
from core.model_handler import ModelHandler
from core.test_model import ModelTester
import pandas as pd

class AILibrary:
    """
    Librería principal para análisis y optimización de código Java.
    """
    def __init__(self, model_path=None):
        """
        Inicializa la librería.

        Args:
            model_path: Ruta al modelo pre-entrenado (opcional)
        """
        self.model_handler = ModelHandler()
        self.trainer = ModelTrainer()
        self.tester = ModelTester()
        self.model = None
        self.metrics = None

        if model_path and os.path.exists(model_path):
            self.model, self.metrics = self.model_handler.load_model(model_path)

    def train(self, data_path, model_type='random_forest'):
        """
        Entrena un nuevo modelo.

        Args:
            data_path: Ruta al archivo CSV con datos de entrenamiento
            model_type: Tipo de modelo ('random_forest' o 'decision_tree')
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

        # Cargar y preparar datos
        training_data = pd.read_csv(data_path)

        # Entrenar modelo
        self.model, self.metrics = self.trainer.train_model(training_data, model_type)
        return self.metrics

    def save_model(self, path):
        """
        Guarda el modelo entrenado.

        Args:
            path: Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar. Entrena o carga un modelo primero.")

        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return self.model_handler.save_model(self.model, path, self.metrics)

    def load_model(self, path):
        """
        Carga un modelo guardado.

        Args:
            path: Ruta al modelo guardado
        """
        self.model, self.metrics = self.model_handler.load_model(path)
        return self.metrics

    def analyze_code(self, metrics_data):
        """
        Analiza métricas de código y predice si es óptimo.

        Args:
            metrics_data: DataFrame con métricas de código
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado. Carga o entrena un modelo primero.")

        results = self.tester.test_model(self.model, metrics_data)
        analysis = self.tester.analyze_predictions(
            metrics_data,
            results['predictions'],
            results['probabilities']
        )

        return results, analysis

    def get_model_metrics(self):
        """
        Obtiene las métricas del modelo actual.
        """
        if self.metrics is None:
            raise ValueError("No hay métricas disponibles. Entrena o carga un modelo primero.")

        return self.metrics

    def get_optimization_suggestions(self, metrics_data, results):
        """
        Genera sugerencias de optimización basadas en el análisis.

        Args:
            metrics_data: DataFrame con métricas de código
            results: Resultados del análisis
        """
        suggestions = []

        for idx, row in metrics_data.iterrows():
            if results['predictions'][idx] == 0:  # Si es subóptimo
                suggestion = {
                    'index': idx,
                    'issues': [],
                    'suggestions': []
                }

                # Verificar diferentes métricas
                if row['cyclomatic_complexity'] > 15:
                    suggestion['issues'].append('Alta complejidad ciclomática')
                    suggestion['suggestions'].append(
                        'Considera dividir el método en funciones más pequeñas'
                    )

                if row['coupling_between_objects'] > 8:
                    suggestion['issues'].append('Alto acoplamiento')
                    suggestion['suggestions'].append(
                        'Considera aplicar principios SOLID para reducir el acoplamiento'
                    )

                if row['lack_of_cohesion'] > 0.5:
                    suggestion['issues'].append('Baja cohesión')
                    suggestion['suggestions'].append(
                        'Considera reorganizar las responsabilidades de la clase'
                    )

                if row['number_of_methods'] > 15:
                    suggestion['issues'].append('Demasiados métodos')
                    suggestion['suggestions'].append(
                        'Considera dividir la clase en clases más pequeñas'
                    )

                if suggestion['issues']:
                    suggestions.append(suggestion)

        return suggestions