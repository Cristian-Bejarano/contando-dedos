import os
import numpy as np

def normalizar_muestra(muestra):
    """Centra los landmarks respecto a la muñeca y los escala según el
    tamaño de la mano, para que el modelo compare solo la FORMA de la mano,
    sin importar su posición en pantalla ni la distancia a la cámara."""

    muestra = muestra.reshape(30, 21, 3)
    muneca = muestra[:, 0:1, :]
    muestra = muestra - muneca
 
    escala = np.linalg.norm(muestra[:, 9, :], axis=1, keepdims=True)
    escala = np.where(escala == 0, 1, escala)
    muestra = muestra / escala[:, np.newaxis, :]
 
    return muestra.reshape(30, 63)

CARPETA_DATOS = "datos"

X = []  # datos o landmarks de cada muestra
y = []  # etiquetas

letras = os.listdir(CARPETA_DATOS)  #['A', 'B', 'C']

for letra in letras:
    carpeta_letra = os.path.join(CARPETA_DATOS, letra)
    archivos = os.listdir(carpeta_letra)  #['0.npy', '1.npy', ...]

    for archivo in archivos:
        ruta = os.path.join(carpeta_letra, archivo)
        muestra = np.load(ruta)  # forma: (30, 63) → 30 frames, 63 números cada uno
        muestra=normalizar_muestra(muestra)

        X.append(muestra)
        y.append(letra)

print(f"Muestras cargadas: {len(X)}")
print(f"Letras encontradas: {set(y)}")

from sklearn.preprocessing import LabelEncoder

X = np.array(X)              # forma: (n_muestras, 30 (frames), 63 (numeros))
X = X.reshape(X.shape[0], -1)  # aplanamos a (numero de muestras, 1890) lo convierte en una sola fila de numeros 'el -1 dice que calcule automaticamente'

label_encoder = LabelEncoder()
y_codificado = label_encoder.fit_transform(y)  # convierte 'A','B','C' a 0, 1, 2 (asi se lleva el arreglo)

print(f"Forma final de X: {X.shape}")
print(f"Clases: {label_encoder.classes_}")

#separar datos para entrenar un modelo

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y_codificado, test_size=0.2, random_state=42
) #80% para train y 20% para test, con ello se hace el modelo predictivo

modelo = RandomForestClassifier(n_estimators=100, random_state=42) #arboles de decision, con ello se muestra cual cree es la respuesta
modelo.fit(X_train, y_train) #modelo de entrenamiento y respuestas correctas

y_pred = modelo.predict(X_test) #pregunta qeu cree que es cada muestra del grupo prueba
precision = accuracy_score(y_test, y_pred) #Compara lo que se predijo vs lo que es en realidad

print(f"Precisión del modelo: {precision * 100:.2f}%")

import joblib
joblib.dump(modelo, "modelo_abecedario.pkl") #Guarda el modelo entrenado
joblib.dump(label_encoder, "label_encoder.pkl") #Guarda la traduccion del modelo 

print("Modelo guardado correctamente")