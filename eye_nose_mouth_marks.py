import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque
import pygame

# ------------------ Initialize Pygame Mixer ------------------ #
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

def play_loop():
    if not pygame.mixer.get_busy():
        alert_sound.play(-1)

def play_drowsy():
    alert_sound.play()
    threading.Timer(2.0, stop_sound).start()

def stop_sound():
    pygame.mixer.stop()

# ------------------ MediaPipe Face Mesh ------------------ #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------ EAR / MAR Functions ------------------ #
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks):
    left = np.array([landmarks[61].x, landmarks[61].y])
    right = np.array([landmarks[291].x, landmarks[291].y])
    horizontal = np.linalg.norm(left - right)
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    vertical = np.linalg.norm(top - bottom)
    return vertical / horizontal

# ------------------ Parameters ------------------ #
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

BLINK_MIN_TIME = 1.0
YAWN_MIN_TIME = 1.0
SLEEP_MIN_TIME = 3.0
NOD_MIN_DELTA = 0.02

BLINK_WINDOW = 2 * 60
YAWN_WINDOW = 5 * 60

BLINK_THRESHOLD = 3
YAWN_THRESHOLD = 3

# ------------------ DSP SMOOTHING ------------------ #
EAR_BUFFER = deque(maxlen=5)
MAR_BUFFER = deque(maxlen=5)

# ------------------ Counters / Timers ------------------ #
blink_times = []
yawn_times = []
nod_times = []

eye_start_time = None
yawn_start_time = None
sleep_start_time = None
sleep_count = 0

nose_y_history = deque(maxlen=10)
nod_in_progress = False

was_sleeping = False
drowsy_alert_playing = False

# ------------------ Camera ------------------ #
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_time = time.time()

    driver_state = "Awake"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # -------- Eyes -------- #
            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]

            ear_raw = (eye_aspect_ratio(right_eye) +
                       eye_aspect_ratio(left_eye)) / 2.0

            EAR_BUFFER.append(ear_raw)
            ear = sum(EAR_BUFFER) / len(EAR_BUFFER)

            # Blink & Sleep detection
            if ear < EAR_THRESHOLD:
                if eye_start_time is None:
                    eye_start_time = current_time
                if sleep_start_time is None:
                    sleep_start_time = current_time
            else:
                if eye_start_time and current_time - eye_start_time >= BLINK_MIN_TIME:
                    blink_times.append(current_time)
                eye_start_time = None

                if was_sleeping:
                    blink_times.clear()
                    yawn_times.clear()
                    nod_times.clear()
                    nose_y_history.clear()
                    nod_in_progress = False
                    was_sleeping = False
                    stop_sound()

                sleep_start_time = None

            if sleep_start_time and (current_time - sleep_start_time) >= SLEEP_MIN_TIME:
                driver_state = "SLEEPING"
                if not was_sleeping:
                    sleep_count += 1
                    was_sleeping = True
                    play_loop()

            blink_times = [t for t in blink_times if current_time - t <= BLINK_WINDOW]

            # -------- Mouth / Yawn -------- #
            mar_raw = mouth_aspect_ratio(face_landmarks.landmark)

            MAR_BUFFER.append(mar_raw)
            mar = sum(MAR_BUFFER) / len(MAR_BUFFER)

            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = current_time
            else:
                if yawn_start_time and current_time - yawn_start_time >= YAWN_MIN_TIME:
                    yawn_times.append(current_time)
                yawn_start_time = None

            yawn_times = [t for t in yawn_times if current_time - t <= YAWN_WINDOW]

            # -------- Head Nod -------- #
            nose_y = face_landmarks.landmark[1].y
            nose_y_history.append(nose_y)

            if len(nose_y_history) >= 2:
                dy = nose_y_history[-1] - nose_y_history[-2]
                if dy > NOD_MIN_DELTA and not nod_in_progress:
                    nod_in_progress = True
                elif dy < -NOD_MIN_DELTA and nod_in_progress:
                    nod_times.append(current_time)
                    nod_in_progress = False

            # -------- Drowsy Detection -------- #
            if driver_state != "SLEEPING":
                if len(blink_times) >= BLINK_THRESHOLD or len(yawn_times) >= YAWN_THRESHOLD:
                    driver_state = "DROWSY"
                    if not drowsy_alert_playing:
                        drowsy_alert_playing = True
                        play_drowsy()
                else:
                    drowsy_alert_playing = False

            # -------- Display Values -------- #
            cv2.putText(frame, f"EAR: {ear:.2f} | Blinks: {len(blink_times)}",
                        (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)
            cv2.putText(frame, f"MAR: {mar:.2f} | Yawns: {len(yawn_times)}",
                        (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)
            cv2.putText(frame, f"Nods: {len(nod_times)}",
                        (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)
            cv2.putText(frame, f"Sleep Count: {sleep_count}",
                        (30,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)

            # -------- LANDMARK INDICATORS (RESTORED) -------- #
            # Eyes (Green)
            for (x, y) in right_eye + left_eye:
                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

            # Mouth + Nose reference (Blue)
            for i in [61, 291, 13, 14, 1]:
                x = int(face_landmarks.landmark[i].x * w)
                y = int(face_landmarks.landmark[i].y * h)
                cv2.circle(frame, (x, y), 2, (255,0,0), -1)

    # -------- Driver State Display -------- #
    color = (0,255,0) if driver_state=="Awake" else \
            (0,255,255) if driver_state=="DROWSY" else (0,0,255)

    cv2.putText(frame, f"Driver State: {driver_state}",
                (30,260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    if driver_state == "DROWSY":
        cv2.putText(frame, "Stay Awake / Take Rest",
                    (30,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,255),2)

    if driver_state == "Awake" and was_sleeping:
        stop_sound()
        was_sleeping = False

    cv2.imshow("Driver Status Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()