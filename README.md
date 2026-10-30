# Lengua de Señas Colombiana

Proyecto de vision por computador para el reconocimiento y traduccion de gestos de LS en tiempo 

---

# Estructura del proyecto 
```text
contando dedos/
├── abecedario/            # Imágenes de entrenamiento/captura
├── datos/                 # Conjuntos de datos estructurados
├── moldes/               # Archivos de modelo y codificadores (.pkl)
├── capturar.py            # Captura de muestras con la cámara
├── menu.py                # Interfaz / Menú principal
├── reconocer_libre.py     # Reconocimiento en tiempo real
├── reconocer_practica.py  # Modo de práctica
├── requerimientos.txt     # Lista de dependencias de Python
└── .gitignore             # Archivos excluidos del control de versiones
```

# Requisitos del sistema
Python: 3.12.10
Recomendado usar entorno virtual (venv)

# Instrucciones de instalacion y uso
git clone <URL_DE_Repositorio>
cd "contando dedos"

# Crear y activar el entorno virtual
python -m venv venv
.\venv\Scripts\avtivate.bat

# Instalar dependencias 
pip install -r requerimientos.txt

# Ejecutar proyecto
python menu.py
