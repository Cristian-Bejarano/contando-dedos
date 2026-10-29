import cv2
import mediapipe as mp
import numpy as np #trabajar datos numericos
import os #crear y trabajar con carpetas
import time #para los timestamps que exige el modo VIDEO

model_path = '../modelos/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,  # VIDEO en vez de IMAGE: usa el frame anterior
    num_hands=1  #solo cumple con una mano           # como contexto, dando un tracking más estable (menos "saltos" de los puntos)
)

LETRA = "Ñ"  # Para cada letra
FRAMES_POR_MUESTRA = 30  # frames que dura
FPS_CUENTA_REGRESIVA = 30  # asumiendo que corre a ~30 fps

cap = cv2.VideoCapture(0)

grabando = False
frames_actuales = []  # Lista para guardar landmarks
contador_regresivo = 0

timestamp_inicio = time.time()  # arranca el reloj para los timestamps del modo VIDEO

# Guarda el último frame donde SÍ se detectó la mano. Arranca en ceros solo
# como respaldo por si el programa empieza sin detectar nada todavía.
# Esto reemplaza el viejo "rellenar con ceros cuando no detecta", que le
# enseñaba al modelo un atajo falso (confundía "ceros" con alguna letra
# específica, en vez de aprender la forma real de la mano).
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

        # El modo VIDEO exige un timestamp creciente en milisegundos
        timestamp_ms = int((time.time() - timestamp_inicio) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        key = cv2.waitKey(1) & 0xFF

        # si espacio y no esta grabando ni hay cuenta regresiva , arranca con la cuenta
        if key == 32 and not grabando and contador_regresivo == 0:
            contador_regresivo = FPS_CUENTA_REGRESIVA * 2  # 2 segundos de cuenta regresiva

        # Mientras la cuenta regresiva esté activa, la vamos bajando
        if contador_regresivo > 0:
            segundos_restantes = contador_regresivo // FPS_CUENTA_REGRESIVA + 1
            cv2.putText(frame, f"Preparate: {segundos_restantes}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            contador_regresivo -= 1

            # Cuando la cuenta regresiva llega a 0, ahí sí empieza a grabar
            if contador_regresivo == 0:
                grabando = True
                frames_actuales = [] 
        # Mientras estemos grabando, guardamos los landmarks de cada frame
        if grabando:
            if result.hand_landmarks:
                mano = result.hand_landmarks[0]  # solo la mano
                landmarks_frame = []
                for lm in mano:
                    landmarks_frame.extend([lm.x, lm.y, lm.z])
                frames_actuales.append(landmarks_frame)
                ultimo_frame_valido = landmarks_frame  # lo guardamos como el último "bueno"
            else:
                # Si en ese frame no detectó la mano, repetimos el último frame
                # válido (en vez de puros ceros) para no enseñarle al modelo
                # un patrón falso de "ausencia de mano"
                frames_actuales.append(ultimo_frame_valido)

            cv2.putText(frame, f"Grabando... {len(frames_actuales)}/{FRAMES_POR_MUESTRA}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # Con suficientes frames se para de grabar
            if len(frames_actuales) >= FRAMES_POR_MUESTRA:
                grabando = False
                print(f"Muestra de '{LETRA}' capturada con {len(frames_actuales)} frames")

            if len(frames_actuales) >= FRAMES_POR_MUESTRA:
                grabando = False

                # Creamos la carpeta de esta letra si no existe
                carpeta_letra = os.path.join("datos", LETRA)
                os.makedirs(carpeta_letra, exist_ok=True)

                # Cuenta cuantas muestras hay
                num_muestras_existentes = len(os.listdir(carpeta_letra))
                nombre_archivo = os.path.join(carpeta_letra, f"{num_muestras_existentes}.npy")

                # Se convierten las listas en un arreglo
                np.save(nombre_archivo, np.array(frames_actuales))

                print(f"Guardado: {nombre_archivo}")

        cv2.imshow("Capturando señas - ESPACIO para grabar, ESC para salir", frame)

        if key == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()