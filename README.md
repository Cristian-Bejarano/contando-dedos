# Lengua de Señas Colombiana

Proyecto de vision por computador para el reconocimiento y traduccion de gestos de LS en tiempo 

---

# Estructura del proyecto 
```text
contando dedos/
├── abcedario/
│   ├── capturar.py
│   ├── entrenar.py
│   ├── evaluar.py
│   ├── inferencia.py
│   ├── reconocer_libre.py
│   ├── reconocer_practica.py
│   ├── datos/          # muestras .npy por letra
│   ├── moldes/         # modelo_abecedario.pkl y label_encoder.pkl
│   └── evaluaciones/   # matrices, campanas y reportes
├── modelos/            # hand_landmarker.task
├── prueba_mediapipe.py
├── requerimientos.txt
└── README.md
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
