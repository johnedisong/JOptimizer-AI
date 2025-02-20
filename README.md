# JOptimizer-AI

Sistema de análisis de código Java que utiliza machine learning para clasificar e identificar oportunidades de optimización. Implementa Random Forest y Árboles de Decisión para determinar si los segmentos de código son óptimos o subóptimos, basándose en métricas de código y datos históricos.

## Características
- Análisis automático de métricas de código Java
- Clasificación mediante Random Forest y Árboles de Decisión
- Evaluación de múltiples métricas de calidad de código
- Identificación de áreas específicas para optimización
- Soporte para diferentes tipos de clases Java (Controllers, Services, Repositories, etc.)

## Requisitos
- Python 3.13 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_del_repositorio]
cd JOptimizer-AI
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Unix o MacOS:
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto
```
JOptimizer-AI/
├── core/                      # Módulos principales
│   ├── model_handler.py      # Gestión de modelos
│   ├── test_model.py         # Pruebas de modelos
│   └── train_model.py        # Entrenamiento
├── utils/                     # Utilidades
│   └── preprocessing.py       # Preprocesamiento de datos
├── data/                      # Datos
│   ├── train/                # Datos de entrenamiento
│   └── test/                 # Datos de prueba
├── models/                    # Modelos entrenados
├── scripts/                   # Scripts ejecutables
│   ├── analyze_data_distribution.py
│   ├── generate_balanced_data.py
│   ├── generate_production_data.py
│   ├── run_production_test.py
│   ├── run_testing.py
│   └── run_training.py
└── config/                    # Configuraciones
    └── parameters.json
```

## Uso

### 1. Generación de Datos de Entrenamiento
```bash
python scripts/generate_balanced_data.py
```
Este script genera un conjunto balanceado de datos de entrenamiento con métricas de código Java.

### 2. Entrenamiento del Modelo
```bash
python python scripts/run_training.py --data data/train/code_metrics.csv --model-type random_forest --output models/mi_modelo.joblib
```
Importante definir siempre un nombre del modelo al cual al final se le va concatenar el timestamp de la hora de ejecución.


Opciones:
- `--model-type`: 'random_forest' o 'decision_tree'
- `--data`: Ruta al archivo CSV de entrenamiento
- `--output`: Ruta donde guardar el modelo entrenado

### 3. Generación de Datos de Prueba
```bash
python scripts/generate_production_data.py
```
Genera datos que simulan métricas de código Java real para pruebas.

### 4. Análisis de Código
```bash
python scripts/run_production_test.py --model ./models/NOMBRE_DEL_MODELO --data ./data/test/production_metrics.csv
```
Analiza código Java y proporciona recomendaciones de optimización.

## Métricas Analizadas
- Lines of Code (LOC)
- Effective Lines of Code
- Cyclomatic Complexity
- Number of Methods
- Inheritance Depth
- Number of Branches
- Coupling Between Objects
- External Dependencies
- Lack of Cohesion

## Interpretación de Resultados
El sistema clasifica el código como:
- **Óptimo**: Código bien estructurado que sigue buenas prácticas
- **Subóptimo**: Código que podría beneficiarse de optimización

Para cada clase analizada, se proporciona:
- Clasificación (Óptimo/Subóptimo)
- Nivel de confianza en la predicción
- Métricas problemáticas específicas
- Recomendaciones de mejora

## Desarrollado por
- Juan Sebastian Fernando Veloza Ceron
- Angelica Daniela Quevedo Cortes
- Alvaro Urrego Viana
- John Edison Goyeneche Barbosa

## Contribuir
Las contribuciones son bienvenidas. Por favor, asegúrese de actualizar las pruebas según corresponda.

## Licencia
Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

### Términos principales:
- ✓ Este software es de uso libre y gratuito
- ✓ Puedes usar el código para cualquier propósito, incluso comercial
- ✓ Puedes modificar, distribuir y sublicenciar el código
- ✓ La única condición es mantener el aviso de copyright y la licencia
- ❗ El software se proporciona "tal cual", sin garantías

Desarrollado como proyecto universitario en la Universidad Internacional de La Rioja (UNIR)