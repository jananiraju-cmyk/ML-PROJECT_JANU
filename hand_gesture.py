import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------
# Recognize fingers
# --------------------------------
def get_fingers(landmarks):
    fingers = []

    # Index, Middle, Ring, Pinky
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    for tip, joint in zip(tips, joints):
        if landmarks[tip].y < landmarks[joint].y:
            fingers.append(1)
        else:
            fingers.append(0)


    return fingers

def recognize_gesture(fingers):

    # [index, middle, ring, pinky]

    if fingers == [0, 0, 0, 0]:
        return "FIST"

    elif fingers == [1, 1, 1, 1]:
        return "OPEN HAND"

    elif fingers == [1, 1, 0, 0]:
        return "PEACE"

    elif fingers == [1, 0, 0, 0]:
        return "POINTING"

    else:
        return "UNKNOWN"


# --------------------------------
# Connections between landmarks
# --------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------
# MediaPipe model setup
# --------------------------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# --------------------------------
# Start webcam
# --------------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started successfully.")
print("Press Q to quit.")


# --------------------------------
# Main loop
# --------------------------------
while True:

    success, frame = camera.read()

    if not success:
        print("Could not read webcam.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp in milliseconds
    timestamp_ms = int(time.monotonic() * 1000)

    # Detect hand
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    gesture = "NO HAND"

    # --------------------------------
    # Hand detected
    # --------------------------------
    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        height, width, _ = frame.shape

        points = []

        # Convert landmarks to screen coordinates
        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        # Draw connections
        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )

        # Draw landmark dots
        for point in points:

            cv2.circle(
                frame,
                point,
                5,
                (0, 255, 0),
                -1
            )

        # Detect fingers
        fingers = get_fingers(landmarks)

        # Recognize gesture
        gesture = recognize_gesture(fingers)


    # --------------------------------
    # Display gesture
    # --------------------------------
    cv2.putText(
        frame,
        gesture,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow(
        "Real-Time Hand Gesture Recognition",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# Cleanup
# --------------------------------
camera.release()
detector.close()
cv2.destroyAllWindows()