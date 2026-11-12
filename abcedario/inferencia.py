import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

CARPETA = "evaluaciones"
os.makedirs(CARPETA, exist_ok=True)

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

resultados = {}
for nombre, norm in [("crudo", False), ("normalizado", True)]:
    X, y = cargar(norm)
    y_cod = LabelEncoder().fit_transform(y)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    acc = cross_val_score(modelo, X, y_cod, cv=cv, n_jobs=-1) * 100
    resultados[nombre] = acc

    media, desv = acc.mean(), acc.std(ddof=1)
    ic = stats.t.interval(0.95, len(acc) - 1, loc=media, scale=stats.sem(acc))
    print(f"{nombre}: media {media:.2f}% | desviación {desv:.2f} | "
          f"IC 95% [{ic[0]:.2f}, {ic[1]:.2f}]")

    plt.figure()
    plt.hist(acc, bins=10, density=True, alpha=0.6)
    x = np.linspace(acc.min() - 2, acc.max() + 2, 200)
    plt.plot(x, stats.norm.pdf(x, media, desv))
    plt.xlabel("Exactitud (%)")
    plt.ylabel("Densidad")
    plt.title(f"Distribución de la exactitud ({nombre})")
    plt.savefig(os.path.join(CARPETA, f"campana_{nombre}.png"), dpi=200)
    plt.close()

t, p = stats.ttest_rel(resultados["normalizado"], resultados["crudo"])
print(f"Prueba t pareada (normalizado vs crudo): t = {t:.3f}, p = {p:.4f}")