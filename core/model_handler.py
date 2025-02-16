# Módulo para gestión de modelos.

import joblib
import os
from datetime import datetime
from typing import Any, Dict

class ModelHandler:
    """
    Clase para manejar el guardado y carga de modelos de machine learning.
    """

    def __init__(self, models_dir: str = "../models"):
        """
        Inicializa el ModelHandler.

        Args:
            models_dir (str): Directorio donde se guardarán/cargarán los modelos
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(self, model: Any, model_name: str, metadata: Dict = None) -> str:
        """
        Guarda un modelo entrenado junto con sus metadatos.

        Args:
            model: Modelo de machine learning entrenado
            model_name (str): Nombre base para el modelo
            metadata (Dict): Información adicional sobre el modelo (opcional)

        Returns:
            str: Ruta donde se guardó el modelo
        """
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = os.path.join(self.models_dir, filename)

        # Preparar metadatos
        if metadata is None:
            metadata = {}

        metadata.update({
            'timestamp': timestamp,
            'model_name': model_name,
            'saved_at': str(datetime.now())
        })

        # Guardar modelo y metadatos
        model_data = {
            'model': model,
            'metadata': metadata
        }

        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")
        return filepath

    def load_model(self, filepath: str) -> tuple:
        """
        Carga un modelo guardado junto con sus metadatos.

        Args:
            filepath (str): Ruta al archivo del modelo

        Returns:
            tuple: (modelo, metadatos)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {filepath}")

        model_data = joblib.load(filepath)
        model = model_data['model']
        metadata = model_data['metadata']

        print(f"Modelo cargado desde: {filepath}")
        print("Metadatos:", metadata)

        return model, metadata

    def list_models(self) -> list:
        """
        Lista todos los modelos guardados en el directorio.

        Returns:
            list: Lista de archivos de modelos disponibles
        """
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                filepath = os.path.join(self.models_dir, filename)
                try:
                    model_data = joblib.load(filepath)
                    metadata = model_data['metadata']
                    models.append({
                        'filename': filename,
                        'filepath': filepath,
                        'metadata': metadata
                    })
                except Exception as e:
                    print(f"Error al cargar {filename}: {str(e)}")
        return models

# Ejemplo de uso
if __name__ == "__main__":
    handler = ModelHandler()

    # Ejemplo de guardado de un modelo dummy
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    metadata = {
        'tipo': 'RandomForest',
        'descripcion': 'Modelo de prueba'
    }

    # Guardar modelo
    filepath = handler.save_model(model, "modelo_prueba", metadata)

    # Listar modelos disponibles
    print("\nModelos disponibles:")
    for model_info in handler.list_models():
        print(f"\nArchivo: {model_info['filename']}")
        print(f"Metadatos: {model_info['metadata']}")

    # Cargar modelo
    loaded_model, loaded_metadata = handler.load_model(filepath)