import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

CARPETA = "evaluaciones"
os.makedirs(CARPETA, exist_ok=True)
DINAMICAS = ["G", "H", "J", "Ñ", "S", "Z"]

def normalizar_muestra(muestra):
    muestra = muestra.reshape(30, 21, 3)
    muestra = muestra - muestra[:, 0:1, :]
    escala = np.linalg.norm(muestra[:, 9, :], axis=1, keepdims=True)
    escala = np.where(escala == 0, 1, escala)
    muestra = muestra / escala[:, np.newaxis, :]
    return muestra.reshape(30, 63)

def cargar(normalizar):
    X, y = [], []
    for letra in sorted(os.listdir("datos")):
        for archivo in sorted(os.listdir(os.path.join("datos", letra))):
            m = np.load(os.path.join("datos", letra, archivo))
            X.append(normalizar_muestra(m) if normalizar else m)
            y.append(letra)
    return np.array(X).reshape(len(X), -1), np.array(y)

for nombre, norm in [("crudo", False), ("normalizado", True)]:
    X, y = cargar(norm)
    le = LabelEncoder()
    y_cod = le.fit_transform(y)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(modelo, X, y_cod, cv=cv)

    cm = confusion_matrix(y_cod, y_pred)
    reporte = classification_report(y_cod, y_pred, target_names=le.classes_, zero_division=0)

    # Acierto medio en letras estáticas y dinámicas
    acierto = cm.diagonal() / cm.sum(axis=1)
    es_dinamica = np.array([l in DINAMICAS for l in le.classes_])
    grupos = (f"Acierto medio en estáticas: {acierto[~es_dinamica].mean() * 100:.1f}% | "
              f"dinámicas: {acierto[es_dinamica].mean() * 100:.1f}%")

    # Las 5 confusiones más frecuentes
    pares = [(cm[i, j], le.classes_[i], le.classes_[j])
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    confundidas = [f"{r} se confundió con {p}: {n} veces"
                   for n, r, p in sorted(pares, reverse=True)[:5]]

    texto = (f"=== {nombre} ===\n"
             f"Exactitud (validación cruzada): {accuracy_score(y_cod, y_pred) * 100:.2f}%\n"
             f"{grupos}\n\n{reporte}\nPares más confundidos:\n" + "\n".join(confundidas))
    print(texto)
    with open(os.path.join(CARPETA, f"reporte_{nombre}.txt"), "w", encoding="utf-8") as f:
        f.write(texto)

    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap="Blues", xticks_rotation=90)
    plt.title(f"Matriz de confusión ({nombre})")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA, f"matriz_{nombre}.png"), dpi=200)
    plt.close()

# Muestras por letra
letras, conteo = np.unique(y, return_counts=True)
plt.figure()
plt.bar(letras, conteo)
plt.title("Muestras por letra")
plt.savefig(os.path.join(CARPETA, "muestras_por_letra.png"), dpi=200)
plt.close()