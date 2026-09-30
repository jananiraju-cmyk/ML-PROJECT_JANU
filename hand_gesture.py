import cv2
import time
import mediapipe as mp

from collections import Counter, deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "hand_landmarker.task"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

SMOOTHING_FRAMES = 7


# ============================================================
# COLORS - OpenCV BGR
# ============================================================

MAROON = (70, 0, 128)
DARK_MAROON = (40, 0, 80)
LIGHT_MAROON = (120, 70, 180)

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (170, 170, 170)

GREEN = (80, 220, 120)
RED = (80, 80, 230)
YELLOW = (80, 210, 255)
BLUE = (255, 180, 80)


# ============================================================
# GESTURE HISTORY
# ============================================================

gesture_history = deque(
    maxlen=SMOOTHING_FRAMES
)


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),

    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (0, 17)
]


# ============================================================
# DETECT FINGERS
# ============================================================

def get_fingers(
    landmarks,
    handedness
):

    """
    Returns:
    [thumb, index, middle, ring, pinky]

    1 = open
    0 = closed
    """

    fingers = []


    # --------------------------------------------------------
    # Thumb
    # --------------------------------------------------------

    thumb_tip = landmarks[4]
    thumb_joint = landmarks[3]

    if handedness == "Right":

        thumb_open = (
            thumb_tip.x
            > thumb_joint.x
        )

    else:

        thumb_open = (
            thumb_tip.x
            < thumb_joint.x
        )

    fingers.append(
        1 if thumb_open else 0
    )


    # --------------------------------------------------------
    # Index, Middle, Ring, Pinky
    # --------------------------------------------------------

    tips = [
        8,
        12,
        16,
        20
    ]

    joints = [
        6,
        10,
        14,
        18
    ]

    for tip, joint in zip(
        tips,
        joints
    ):

        if (
            landmarks[tip].y
            < landmarks[joint].y
        ):

            fingers.append(1)

        else:

            fingers.append(0)


    return fingers


# ============================================================
# RECOGNIZE GESTURE
# ============================================================

def recognize_gesture(
    fingers,
    landmarks
):

    # fingers =
    # [thumb, index, middle, ring, pinky]


    # --------------------------------------------------------
    # FIST
    # --------------------------------------------------------

    if fingers == [
        0, 0, 0, 0, 0
    ]:

        return "FIST"


    # --------------------------------------------------------
    # OPEN HAND
    # --------------------------------------------------------

    elif fingers == [
        1, 1, 1, 1, 1
    ]:

        return "OPEN HAND"


    # --------------------------------------------------------
    # PEACE
    # --------------------------------------------------------

    elif fingers == [
        0, 1, 1, 0, 0
    ]:

        return "PEACE"


    # --------------------------------------------------------
    # POINTING
    # --------------------------------------------------------

    elif fingers == [
        0, 1, 0, 0, 0
    ]:

        return "POINTING"


    # --------------------------------------------------------
    # THREE
    # --------------------------------------------------------

    elif fingers == [
        0, 1, 1, 1, 0
    ]:

        return "THREE"


    # --------------------------------------------------------
    # ROCK
    # --------------------------------------------------------

    elif (
        fingers[1] == 1
        and fingers[4] == 1
        and fingers[2] == 0
        and fingers[3] == 0
    ):

        return "ROCK"


    # --------------------------------------------------------
    # THUMBS UP / DOWN
    # --------------------------------------------------------

    elif (
        fingers[0] == 1
        and fingers[1:] == [
            0, 0, 0, 0
        ]
    ):

        thumb_tip_y = landmarks[4].y
        wrist_y = landmarks[0].y

        if thumb_tip_y < wrist_y:

            return "THUMBS UP"

        else:

            return "THUMBS DOWN"


    return "UNKNOWN"


# ============================================================
# SMOOTH GESTURE
# ============================================================

def smooth_gesture(
    current_gesture
):

    gesture_history.append(
        current_gesture
    )

    counter = Counter(
        gesture_history
    )

    return counter.most_common(
        1
    )[0][0]


# ============================================================
# GESTURE COLOR
# ============================================================

def get_gesture_color(
    gesture
):

    colors = {

        "FIST": RED,

        "OPEN HAND": GREEN,

        "PEACE": YELLOW,

        "POINTING": BLUE,

        "THREE": LIGHT_MAROON,

        "ROCK": RED,

        "THUMBS UP": GREEN,

        "THUMBS DOWN": RED,

        "UNKNOWN": GRAY,

        "NO HAND": GRAY
    }

    return colors.get(
        gesture,
        WHITE
    )


# ============================================================
# ROUNDED RECTANGLE
# ============================================================

def rounded_rectangle(
    image,
    top_left,
    bottom_right,
    color,
    radius=20
):

    x1, y1 = top_left
    x2, y2 = bottom_right


    cv2.rectangle(
        image,
        (
            x1 + radius,
            y1
        ),
        (
            x2 - radius,
            y2
        ),
        color,
        -1
    )


    cv2.rectangle(
        image,
        (
            x1,
            y1 + radius
        ),
        (
            x2,
            y2 - radius
        ),
        color,
        -1
    )


    cv2.circle(
        image,
        (
            x1 + radius,
            y1 + radius
        ),
        radius,
        color,
        -1
    )


    cv2.circle(
        image,
        (
            x2 - radius,
            y1 + radius
        ),
        radius,
        color,
        -1
    )


    cv2.circle(
        image,
        (
            x1 + radius,
            y2 - radius
        ),
        radius,
        color,
        -1
    )


    cv2.circle(
        image,
        (
            x2 - radius,
            y2 - radius
        ),
        radius,
        color,
        -1
    )


# ============================================================
# TRANSPARENT PANEL
# ============================================================

def draw_panel(
    frame,
    top_left,
    bottom_right,
    color,
    alpha=0.75,
    radius=20
):

    overlay = frame.copy()


    rounded_rectangle(
        overlay,
        top_left,
        bottom_right,
        color,
        radius
    )


    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.HandLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.6,

    min_hand_presence_confidence=0.6,

    min_tracking_confidence=0.6
)


detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# START CAMERA
# ============================================================

camera = cv2.VideoCapture(
    0
)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)


camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


if not camera.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    detector.close()

    exit()


# ============================================================
# CREATE LARGE WINDOW
# ============================================================

WINDOW_NAME = (
    "AI Hand Gesture Recognition"
)


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    WINDOW_NAME,
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)


print()
print(
    "========================================="
)
print(
    "       AI Hand Gesture Recognition"
)
print(
    "========================================="
)
print(
    "Camera started successfully."
)
print(
    "Resolution: 1280 x 720"
)
print(
    "Press Q to quit."
)
print()


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = time.time()

fps = 0

previous_timestamp = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "Could not read webcam."
        )

        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # FPS calculation
    # --------------------------------------------------------

    current_time = time.time()

    time_difference = (
        current_time
        - previous_time
    )


    if time_difference > 0:

        current_fps = (
            1 / time_difference
        )

        fps = (
            0.9 * fps
            + 0.1 * current_fps
        )


    previous_time = current_time


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Convert to MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms = int(
        time.monotonic()
        * 1000
    )


    if (
        timestamp_ms
        <= previous_timestamp
    ):

        timestamp_ms = (
            previous_timestamp
            + 1
        )


    previous_timestamp = timestamp_ms


    # --------------------------------------------------------
    # Hand detection
    # --------------------------------------------------------

    result = detector.detect_for_video(

        mp_image,

        timestamp_ms
    )


    gesture = "NO HAND"

    handedness_text = ""

    confidence_text = ""


    # ========================================================
    # HAND FOUND
    # ========================================================

    if result.hand_landmarks:

        landmarks = (
            result.hand_landmarks[0]
        )


        # ----------------------------------------------------
        # Handedness
        # ----------------------------------------------------

        if result.handedness:

            category = (
                result.handedness[0][0]
            )


            handedness_text = (
                category.category_name
            )


            confidence_text = (
                f"{category.score * 100:.1f}%"
            )


        # ----------------------------------------------------
        # Convert landmarks to pixels
        # ----------------------------------------------------

        points = []


        for landmark in landmarks:

            px = int(
                landmark.x
                * width
            )


            py = int(
                landmark.y
                * height
            )


            points.append(
                (px, py)
            )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x_values = [
            point[0]
            for point in points
        ]


        y_values = [
            point[1]
            for point in points
        ]


        box_x1 = max(
            min(x_values) - 30,
            0
        )


        box_y1 = max(
            min(y_values) - 30,
            0
        )


        box_x2 = min(
            max(x_values) + 30,
            width
        )


        box_y2 = min(
            max(y_values) + 30,
            height
        )


        # ----------------------------------------------------
        # Main bounding rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (
                box_x1,
                box_y1
            ),
            (
                box_x2,
                box_y2
            ),
            MAROON,
            2
        )


        # ----------------------------------------------------
        # Stylish corners
        # ----------------------------------------------------

        corner_length = 35

        corner_thickness = 5


        # Top Left
        cv2.line(
            frame,
            (
                box_x1,
                box_y1
            ),
            (
                box_x1 + corner_length,
                box_y1
            ),
            WHITE,
            corner_thickness
        )


        cv2.line(
            frame,
            (
                box_x1,
                box_y1
            ),
            (
                box_x1,
                box_y1 + corner_length
            ),
            WHITE,
            corner_thickness
        )


        # Top Right
        cv2.line(
            frame,
            (
                box_x2,
                box_y1
            ),
            (
                box_x2 - corner_length,
                box_y1
            ),
            WHITE,
            corner_thickness
        )


        cv2.line(
            frame,
            (
                box_x2,
                box_y1
            ),
            (
                box_x2,
                box_y1 + corner_length
            ),
            WHITE,
            corner_thickness
        )


        # Bottom Left
        cv2.line(
            frame,
            (
                box_x1,
                box_y2
            ),
            (
                box_x1 + corner_length,
                box_y2
            ),
            WHITE,
            corner_thickness
        )


        cv2.line(
            frame,
            (
                box_x1,
                box_y2
            ),
            (
                box_x1,
                box_y2 - corner_length
            ),
            WHITE,
            corner_thickness
        )


        # Bottom Right
        cv2.line(
            frame,
            (
                box_x2,
                box_y2
            ),
            (
                box_x2 - corner_length,
                box_y2
            ),
            WHITE,
            corner_thickness
        )


        cv2.line(
            frame,
            (
                box_x2,
                box_y2
            ),
            (
                box_x2,
                box_y2 - corner_length
            ),
            WHITE,
            corner_thickness
        )


        # ----------------------------------------------------
        # Draw hand skeleton
        # ----------------------------------------------------

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                LIGHT_MAROON,
                3,
                cv2.LINE_AA
            )


        # ----------------------------------------------------
        # Draw landmark points
        # ----------------------------------------------------

        for index, point in enumerate(
            points
        ):

            # Fingertips
            if index in [
                4,
                8,
                12,
                16,
                20
            ]:

                cv2.circle(
                    frame,
                    point,
                    9,
                    WHITE,
                    -1
                )


                cv2.circle(
                    frame,
                    point,
                    6,
                    MAROON,
                    -1
                )


            else:

                cv2.circle(
                    frame,
                    point,
                    5,
                    WHITE,
                    -1
                )


        # ----------------------------------------------------
        # Detect finger states
        # ----------------------------------------------------

        fingers = get_fingers(

            landmarks,

            handedness_text
        )


        # ----------------------------------------------------
        # Recognize gesture
        # ----------------------------------------------------

        raw_gesture = recognize_gesture(

            fingers,

            landmarks
        )


        # ----------------------------------------------------
        # Smooth gesture
        # ----------------------------------------------------

        gesture = smooth_gesture(
            raw_gesture
        )


    else:

        gesture_history.clear()


    # ========================================================
    # TOP HEADER PANEL
    # ========================================================

    draw_panel(
        frame,
        (
            25,
            20
        ),
        (
            width - 25,
            110
        ),
        DARK_MAROON,
        alpha=0.82,
        radius=20
    )


    cv2.putText(
        frame,
        "AI HAND GESTURE RECOGNITION",
        (
            50,
            60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "MediaPipe + OpenCV",
        (
            50,
            90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (
            220,
            220,
            220
        ),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # FPS DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",
        (
            width - 180,
            72
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # BOTTOM PANEL
    # ========================================================

    gesture_color = get_gesture_color(
        gesture
    )


    panel_height = 125


    panel_y1 = (
        height
        - panel_height
        - 20
    )


    panel_y2 = (
        height
        - 20
    )


    draw_panel(
        frame,
        (
            25,
            panel_y1
        ),
        (
            width - 25,
            panel_y2
        ),
        BLACK,
        alpha=0.8,
        radius=25
    )


    # ========================================================
    # GESTURE LABEL
    # ========================================================

    cv2.putText(
        frame,
        "DETECTED GESTURE",
        (
            55,
            panel_y1 + 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GRAY,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        gesture,
        (
            55,
            panel_y1 + 88
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        gesture_color,
        3,
        cv2.LINE_AA
    )


    # ========================================================
    # HAND INFORMATION
    # ========================================================

    if handedness_text:

        cv2.putText(
            frame,
            f"{handedness_text} Hand",
            (
                width - 300,
                panel_y1 + 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            WHITE,
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Confidence: {confidence_text}",
            (
                width - 300,
                panel_y1 + 80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            GRAY,
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # QUIT MESSAGE
    # ========================================================

    cv2.putText(
        frame,
        "Press Q to quit",
        (
            width - 170,
            height - 5
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        GRAY,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(
        1
    ) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

detector.close()

cv2.destroyAllWindows()

print()
print(
    "Program stopped."
)
