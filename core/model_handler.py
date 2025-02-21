import joblib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

class ModelHandler:
    """
    Clase para manejar el guardado y carga de modelos de machine learning.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Inicializa el ModelHandler.

        Args:
            models_dir (str): Directorio donde se guardarán/cargarán los modelos
        """
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = self.models_dir / filename

        if metadata is None:
            metadata = {}

        metadata.update({
            'timestamp': timestamp,
            'model_name': model_name,
            'saved_at': str(datetime.now())
        })

        model_data = {
            'model': model,
            'metadata': metadata
        }

        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")
        return str(filepath)

    def load_model(self, filepath: str) -> tuple:
        """
        Carga un modelo guardado junto con sus metadatos.

        Args:
            filepath (str): Ruta al archivo del modelo

        Returns:
            tuple: (modelo, metadatos)
        """
        filepath = Path(filepath).resolve()
        if not filepath.exists():
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
        for filepath in self.models_dir.glob("*.joblib"):
            try:
                model_data = joblib.load(filepath)
                metadata = model_data['metadata']
                models.append({
                    'filename': filepath.name,
                    'filepath': str(filepath),
                    'metadata': metadata
                })
            except Exception as e:
                print(f"Error al cargar {filepath.name}: {str(e)}")
        return models

# Ejemplo de uso
if __name__ == "__main__":
    handler = ModelHandler()

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    metadata = {
        'tipo': 'RandomForest',
        'descripcion': 'Modelo de prueba'
    }

    filepath = handler.save_model(model, "modelo_prueba", metadata)

    print("\nModelos disponibles:")
    for model_info in handler.list_models():
        print(f"\nArchivo: {model_info['filename']}")
        print(f"Metadatos: {model_info['metadata']}")

    loaded_model, loaded_metadata = handler.load_model(filepath)