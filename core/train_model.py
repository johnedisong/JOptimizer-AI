# Módulo para entrenamiento de modelos.

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

class ModelTrainer:
    """
    Clase para entrenar y evaluar modelos de clasificación de código.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa el entrenador de modelos.

        Args:
            random_state (int): Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.models = {
            'random_forest': RandomForestClassifier(random_state=random_state),
            'decision_tree': DecisionTreeClassifier(random_state=random_state)
        }

    def train_model(self,
                   data: pd.DataFrame,
                   model_type: str = 'random_forest',
                   test_size: float = 0.2) -> Tuple[Any, Dict]:
        """
        Entrena un modelo con los datos proporcionados.

        Args:
            data (pd.DataFrame): DataFrame con las métricas y etiquetas
            model_type (str): Tipo de modelo ('random_forest' o 'decision_tree')
            test_size (float): Proporción de datos para pruebas

        Returns:
            Tuple[Any, Dict]: Modelo entrenado y métricas de evaluación
        """
        if model_type not in self.models:
            raise ValueError(f"Tipo de modelo no válido. Opciones: {list(self.models.keys())}")

        # Separar características y etiquetas
        X = data.drop('is_optimal', axis=1)
        y = data['is_optimal']

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Entrenar modelo
        model = self.models[model_type]
        model.fit(X_train, y_train)

        # Evaluar modelo
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Predicciones
        y_pred = model.predict(X_test)

        # Métricas de evaluación
        metrics = {
            'train_score': train_score,
            'test_score': test_score,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        if model_type == 'random_forest':
            # Importancia de características para Random Forest
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            metrics['feature_importance'] = feature_importance

        return model, metrics

    def get_model_params(self, model_type: str) -> Dict:
        """
        Obtiene los parámetros actuales del modelo.

        Args:
            model_type (str): Tipo de modelo

        Returns:
            Dict: Parámetros del modelo
        """
        if model_type not in self.models:
            raise ValueError(f"Tipo de modelo no válido. Opciones: {list(self.models.keys())}")

        return self.models[model_type].get_params()

    def set_model_params(self, model_type: str, params: Dict):
        """
        Establece nuevos parámetros para el modelo.

        Args:
            model_type (str): Tipo de modelo
            params (Dict): Nuevos parámetros
        """
        if model_type not in self.models:
            raise ValueError(f"Tipo de modelo no válido. Opciones: {list(self.models.keys())}")

        self.models[model_type].set_params(**params)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos de ejemplo
    from utils.preprocessing import analyze_code_metrics

    data, _ = analyze_code_metrics('../data/train/code_metrics.csv')

    # Crear entrenador
    trainer = ModelTrainer()

    # Entrenar modelo Random Forest
    rf_model, rf_metrics = trainer.train_model(data, 'random_forest')

    print("\n=== Métricas del Modelo Random Forest ===")
    print(f"Precisión en entrenamiento: {rf_metrics['train_score']:.3f}")
    print(f"Precisión en pruebas: {rf_metrics['test_score']:.3f}")

    print("\nImportancia de características:")
    for feature, importance in sorted(
        rf_metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{feature}: {importance:.3f}")

    # Entrenar modelo Decision Tree
    dt_model, dt_metrics = trainer.train_model(data, 'decision_tree')

    print("\n=== Métricas del Modelo Decision Tree ===")
    print(f"Precisión en entrenamiento: {dt_metrics['train_score']:.3f}")
    print(f"Precisión en pruebas: {dt_metrics['test_score']:.3f}")