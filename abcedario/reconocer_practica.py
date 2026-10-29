import cv2
import mediapipe as mp
import numpy as np
import joblib

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

# Cargamos el modelo ya entrenado y su "traductor" de números a letras
modelo = joblib.load("modelo_abecedario.pkl")
label_encoder = joblib.load("label_encoder.pkl")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)

FRAMES_POR_MUESTRA = 30
grabando = False
frames_actuales = []
contador_regresivo = 0

import time
timestamp_inicio=time.time()
letra_predicha = ""  #Variable qeu va a guardar la prediccion del modelo

# Guarda el último frame donde SÍ se detectó la mano. Arranca en ceros solo
# como respaldo inicial. Reemplaza el viejo "rellenar con puros ceros cuando
# no detecta", que le enseñaba al modelo un atajo falso (aprendía a asociar
# "muchos ceros seguidos" con una letra específica, en vez de aprender la
# forma real de la mano) — por eso se confundía P/N/Ñ cuando se perdía el
# tracking en vez de cuando de verdad veía esas señas.
ultimo_frame_valido = [0.0] * 63

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms=int((time.time()-timestamp_inicio)*1000)
        result= landmarker.detect_for_video(mp_image, timestamp_ms)

        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Dibujar las conexiones (líneas verdes)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    p1 = hand_landmarks[start_idx]
                    p2 = hand_landmarks[end_idx]
                    x1, y1 = int(p1.x * width), int(p1.y * height)
                    x2, y2 = int(p2.x * width), int(p2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Dibujar las articulaciones (puntos azules)
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        #Misma logica de capturar, lo ideal es agrupar y comprimir los30 frames

        key = cv2.waitKey(1) & 0xFF

        if key == 32 and not grabando and contador_regresivo == 0:
            contador_regresivo = 30 * 2  # 2 segundos de cuenta regresiva

        if contador_regresivo > 0:
            cv2.putText(frame, f"Preparate {contador_regresivo // 30 + 1}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            contador_regresivo -= 1
            if contador_regresivo == 0:
                grabando = True
                frames_actuales = []

        if grabando:
            if result.hand_landmarks:
                mano = result.hand_landmarks[0]
                landmarks_frame = []
                for lm in mano:
                    landmarks_frame.extend([lm.x, lm.y, lm.z])
                frames_actuales.append(landmarks_frame)
                ultimo_frame_valido = landmarks_frame  # lo guardamos como el último "bueno"
            else:
                # Repetimos el último frame válido en vez de ceros
                frames_actuales.append(ultimo_frame_valido)

            cv2.putText(frame, f"Reconociendo... {len(frames_actuales)}/{FRAMES_POR_MUESTRA}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            #Cuando se junten los frmes pregunar al modelo

            if len(frames_actuales) >= FRAMES_POR_MUESTRA:
                grabando = False

                # Aplanamos la secuencia igual que hicimos en entrenar.py
                muestra = np.array(frames_actuales)          # forma: (30, 63)
                muestra=normalizar_muestra(muestra)
                muestra = muestra.reshape(1, -1)              # forma: (1, 1890), esto traduce que el modelo solo espera recibir una tabla de muestras

                # Le pregunta al modelo
                prediccion_numero = modelo.predict(muestra)[0] #predice el resultado y lo devuelve como unico
                letra_predicha = label_encoder.inverse_transform([prediccion_numero])[0] #convertimos el numero a letra

                print(f"La letra es: {letra_predicha}")

        # Si ya tenemos una predicción, la mostramos en pantalla
        if letra_predicha:
            cv2.putText(frame, f"Letra: {letra_predicha}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow("Leen-Señas - Reconocimiento con practica", frame)

        if key == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()