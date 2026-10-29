import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import time

model_path = '../modelos/hand_landmarker.task'

def normalizar_muestra(muestra):
    muestra = muestra.reshape(30, 21, 3)
    muneca = muestra[:, 0:1, :]
    muestra = muestra - muneca
 
    escala = np.linalg.norm(muestra[:, 9, :], axis=1, keepdims=True)
    escala = np.where(escala == 0, 1, escala)
    muestra = muestra / escala[:, np.newaxis, :]
 
    return muestra.reshape(30, 63)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

# Cargamos el modelo entrenado y el encoder de etiquetas
modelo = joblib.load("modelo_abecedario.pkl")
label_encoder = joblib.load("label_encoder.pkl")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)

FRAMES_POR_MUESTRA = 30
buffer_frames = deque(maxlen=FRAMES_POR_MUESTRA)
letra_predicha = ""

timestamp_inicio = time.time()

# Ajustes de consenso ultra-rápidos
TAMANO_HISTORIAL = 5       # Evaluamos solo los últimos 5 cuadros predecidos
VOTOS_REQUERIDOS = 3       # Con que 3 de 5 coincidan, se confirma la seña
UMBRAL_CONFIANZA = 0.50    # Permisivo para transiciones naturales (50% de certeza)

historial_predicciones = deque(maxlen=TAMANO_HISTORIAL)
letra_mostrada = ""

pausado = False
FRAMES_PAUSA = 15          # ~0.5 segundos de pausa tras detectar antes de recibir otra seña
contador_pausa = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - timestamp_inicio) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Dibujar landmarks en pantalla
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for start_idx, end_idx in HAND_CONNECTIONS:
                    p1 = hand_landmarks[start_idx]
                    p2 = hand_landmarks[end_idx]
                    x1, y1 = int(p1.x * width), int(p1.y * height)
                    x2, y2 = int(p2.x * width), int(p2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Control de flujo y predicción
        if pausado:
            contador_pausa += 1
            if contador_pausa >= FRAMES_PAUSA:
                pausado = False
                contador_pausa = 0
                buffer_frames.clear()
                historial_predicciones.clear()
        else:
            if result.hand_landmarks:
                mano = result.hand_landmarks[0]
                landmarks_frame = []
                for lm in mano:
                    landmarks_frame.extend([lm.x, lm.y, lm.z])
                buffer_frames.append(landmarks_frame)
            else:
                # Si se quita la mano, reseteamos para limpiar el estado
                buffer_frames.clear()
                historial_predicciones.clear()

            # Predicción continua cuando el buffer de 30 frames se llena
            if len(buffer_frames) == FRAMES_POR_MUESTRA:
                muestra = np.array(buffer_frames)
                muestra = normalizar_muestra(muestra)
                muestra = muestra.reshape(1, -1)
                
                # Evaluación de probabilidades
                probas = modelo.predict_proba(muestra)[0]
                max_proba = np.max(probas)
                prediccion_numero = np.argmax(probas)

                # Solo agregamos al historial si pasa el umbral (sin borrar el historial previo si falla)
                if max_proba >= UMBRAL_CONFIANZA:
                    letra_predicha = label_encoder.inverse_transform([prediccion_numero])[0]
                    historial_predicciones.append(letra_predicha)

                # Si tenemos suficientes predicciones acumuladas, evaluamos consenso
                if len(historial_predicciones) >= VOTOS_REQUERIDOS:
                    letra_mas_comun, veces = Counter(historial_predicciones).most_common(1)[0]
                    if veces >= VOTOS_REQUERIDOS:
                        letra_mostrada = letra_mas_comun
                        pausado = True
                        contador_pausa = 0

        key = cv2.waitKey(1) & 0xFF

        # Interfaz visual
        if letra_mostrada:
            cv2.putText(frame, f"Letra: {letra_mostrada}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        if pausado:
            cv2.putText(frame, "Reconocido - espera...", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Listo - haz tu seña", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Leen-Señas - Reconocimiento en vivo", frame)

        if key == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()