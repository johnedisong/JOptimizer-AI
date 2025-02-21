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

## Configuración de Algoritmos de Machine Learning

A continuación se describen los parámetros configurables para los algoritmos de Machine Learning utilizados en el entrenamiento de modelos.

📌 **Ubicación de la configuración:**
Los parámetros se encuentran en el archivo `parameters.json`, ubicado en la carpeta `config/` dentro del proyecto.

```
config/
│── parameters.json
```

## 📌 **Modelos Disponibles**
### 🔹 **Random Forest**
Parámetros ajustables para el modelo `RandomForestClassifier`:
- **`n_estimators`** *(int)* → Número de árboles en el bosque. *(Por defecto: 100)*
- **`max_depth`** *(int or None)* → Profundidad máxima de cada árbol. *(Por defecto: 10)*
- **`min_samples_split`** *(int)* → Número mínimo de muestras requeridas para dividir un nodo. *(Por defecto: 2)*
- **`min_samples_leaf`** *(int)* → Número mínimo de muestras requeridas en una hoja. *(Por defecto: 1)*

### 🔹 **Decision Tree**
Parámetros ajustables para el modelo `DecisionTreeClassifier`:
- **`max_depth`** *(int or None)* → Profundidad máxima del árbol. *(Por defecto: 10)*
- **`min_samples_split`** *(int)* → Número mínimo de muestras requeridas para dividir un nodo. *(Por defecto: 2)*
- **`min_samples_leaf`** *(int)* → Número mínimo de muestras requeridas en una hoja. *(Por defecto: 1)*

---

## ⚙ **Configuración de Entrenamiento**
- **`test_size`** *(float)* → Proporción de los datos reservados para pruebas. *(Por defecto: 0.2)*
- **`random_state`** *(int)* → Semilla para la reproducción de resultados. *(Por defecto: 42)*

---

## 🔄 **Preprocesamiento de Datos**
- **`normalize`** *(bool)* → Indica si se debe normalizar los datos antes del entrenamiento. *(Por defecto: `true`)*
- **`remove_outliers`** *(bool)* → Indica si se deben eliminar valores atípicos. *(Por defecto: `true`)*

### 📌 **Notas**
- Estos valores pueden ser modificados en el archivo `config/parameters.json` según las necesidades del entrenamiento.
- La configuración afecta la calidad y velocidad del modelo entrenado.
- Se recomienda experimentar con distintos valores para optimizar el rendimiento.

🚀 **Modifica estos parámetros en el archivo de configuración para ajustar el comportamiento del modelo según tus necesidades.**



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
por ejemplo 

```bash
python scripts/run_production_test.py --model ./models/mi_modelo444_20250220_200717.joblib --data ./data/test/production_metrics.csv
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