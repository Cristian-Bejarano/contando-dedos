import cv2
import mediapipe as mp

# 1. Configuración de MediaPipe Tasks
model_path = 'modelos/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Conexiones de los 21 puntos para dibujar la mano
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),           # Pulgar
    (0,5), (5,6), (6,7), (7,8),           # Índice
    (5,9), (9,10), (10,11), (11,12),      # Medio
    (9,13), (13,14), (14,15), (15,16),    # Anular
    (13,17), (17,18), (18,19), (19,20),   # Meñique
    (0,17)                                # Palma
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

cap = cv2.VideoCapture(0)

# 2. Bucle principal de detección
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Convertir la imagen al formato que requiere MediaPipe Tasks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Procesar frame
        result = landmarker.detect(mp_image)

        # Dibujar los puntos y líneas si hay detección
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

        cv2.imshow("Mano detectada - MediaPipe Tasks", frame)

        # Presiona ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()