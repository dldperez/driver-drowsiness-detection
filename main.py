import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading
import time

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

def mouth_aspect_ratio_norm(landmarks):
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

BLINK_WINDOW = 2 * 60  # 2 minutes
YAWN_WINDOW = 5 * 60   # 5 minutes

BLINK_THRESHOLD = 3
YAWN_THRESHOLD = 3

FRAME_RATE = 30  # camera FPS
BLINK_MIN_FRAMES = 2 * FRAME_RATE  # 2 seconds
YAWN_MIN_FRAMES = 2 * FRAME_RATE   # 2 seconds

# ------------------ Counters ------------------ #
eye_frame_counter = 0
yawn_frame_counter = 0
blink_times = []
yawn_times = []

# Countdown timers
blink_window_start = None
yawn_window_start = None

# ------------------ Alert ------------------ #
ALERT_SOUND = "alert.wav"
last_alert_time = 0
ALERT_COOLDOWN = 5

def play_alert():
    threading.Thread(target=playsound, args=(ALERT_SOUND,), daemon=True).start()

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

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # -------- Eyes -------- #
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            ear = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2.0

            # Detect blink only if eyes closed for 2 seconds
            if ear < EAR_THRESHOLD:
                eye_frame_counter += 1
            else:
                if eye_frame_counter >= BLINK_MIN_FRAMES:
                    blink_times.append(current_time)
                    if blink_window_start is None:
                        blink_window_start = current_time
                eye_frame_counter = 0

            # Remove old blink events
            blink_times = [t for t in blink_times if current_time - t <= BLINK_WINDOW]
            blink_window_start = blink_times[0] if blink_times else None

            # -------- Mouth / Yawn -------- #
            mar = mouth_aspect_ratio_norm(face_landmarks.landmark)
            if mar > MAR_THRESHOLD:
                yawn_frame_counter += 1
            else:
                if yawn_frame_counter >= YAWN_MIN_FRAMES:
                    yawn_times.append(current_time)
                    if yawn_window_start is None:
                        yawn_window_start = current_time
                yawn_frame_counter = 0

            # Remove old yawn events
            yawn_times = [t for t in yawn_times if current_time - t <= YAWN_WINDOW]
            yawn_window_start = yawn_times[0] if yawn_times else None

            # -------- Display EAR / MAR -------- #
            cv2.putText(frame, f"EAR: {ear:.2f} | Blinks: {len(blink_times)}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f} | Yawns: {len(yawn_times)}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # -------- Display Countdown Timers -------- #
            if blink_window_start:
                blink_countdown = max(0, int(BLINK_WINDOW - (current_time - blink_window_start)))
                cv2.putText(frame, f"Blink Window: {blink_countdown}s", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if yawn_window_start:
                yawn_countdown = max(0, int(YAWN_WINDOW - (current_time - yawn_window_start)))
                cv2.putText(frame, f"Yawn Window: {yawn_countdown}s", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # -------- Draw Landmarks -------- #
            for (x, y) in right_eye + left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for i in [61, 291, 13, 14]:
                x, y = int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # -------- Danger Alert -------- #
    danger = len(blink_times) >= BLINK_THRESHOLD or len(yawn_times) >= YAWN_THRESHOLD
    if danger:
        cv2.putText(frame, "!!! DANGER ALERT !!!", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if (current_time - last_alert_time) > ALERT_COOLDOWN:
            play_alert()
            last_alert_time = current_time

    cv2.imshow("Driver Status Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()