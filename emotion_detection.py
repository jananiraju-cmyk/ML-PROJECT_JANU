import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------
# 1. Expression labels
# --------------------------------------
EXPRESSIONS = [
    "ANGRY",
    "DISGUST",
    "FEARFUL",
    "HAPPY",
    "NEUTRAL",
    "SAD",
    "SURPRISED"
]


# --------------------------------------
# 2. Japanese expression labels
# --------------------------------------
JAPANESE_EXPRESSIONS = {
    "ANGRY": "怒り",
    "DISGUST": "嫌悪",
    "FEARFUL": "恐怖",
    "HAPPY": "嬉しい",
    "NEUTRAL": "普通",
    "SAD": "悲しい",
    "SURPRISED": "驚き"
}


# --------------------------------------
# 3. Ask user's name
# --------------------------------------
person_name = input("Enter your name: ").strip()

if not person_name:
    person_name = "User"


# --------------------------------------
# 4. Model files
# --------------------------------------
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL = "facial_expression_recognition.onnx"


# --------------------------------------
# 5. Japanese font
# --------------------------------------
FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"

japanese_font = ImageFont.truetype(
    FONT_PATH,
    26
)


# --------------------------------------
# 6. Stylish text colors
# Pillow uses RGB
# --------------------------------------
TEXT_COLOR = (128, 0, 0)         # Maroon
OUTLINE_COLOR = (255, 255, 255)  # White
BOX_COLOR = (255, 240, 245)      # Light pink


# --------------------------------------
# 7. Face rectangle color
# OpenCV uses BGR
# --------------------------------------
FACE_BOX_COLOR = (80, 0, 128)    # Maroon-ish
LANDMARK_COLOR = (255, 180, 0)


# --------------------------------------
# 8. Standard face landmark positions
# --------------------------------------
STANDARD_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)


# --------------------------------------
# 9. Create face detector
# --------------------------------------
face_detector = cv2.FaceDetectorYN_create(
    FACE_MODEL,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


# --------------------------------------
# 10. Load expression model
# --------------------------------------
emotion_net = cv2.dnn.readNetFromONNX(
    EMOTION_MODEL
)

emotion_net.setPreferableBackend(
    cv2.dnn.DNN_BACKEND_OPENCV
)

emotion_net.setPreferableTarget(
    cv2.dnn.DNN_TARGET_CPU
)


# --------------------------------------
# 11. Align face using 5 landmarks
# --------------------------------------
def align_face(frame, landmarks):

    landmarks = landmarks.astype(
        np.float32
    )

    transform, _ = cv2.estimateAffinePartial2D(
        landmarks,
        STANDARD_POINTS,
        method=cv2.LMEDS
    )

    if transform is None:
        return None

    aligned = cv2.warpAffine(
        frame,
        transform,
        (112, 112)
    )

    return aligned


# --------------------------------------
# 12. Classify facial expression
# --------------------------------------
def classify_expression(face):

    # BGR -> RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Convert 0-255 to 0-1
    face = face.astype(
        np.float32
    ) / 255.0

    # Normalize to approximately -1 to 1
    face = (face - 0.5) / 0.5

    # HWC -> CHW
    face = np.transpose(
        face,
        (2, 0, 1)
    )

    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    emotion_net.setInput(
        face,
        "data"
    )

    output = emotion_net.forward(
        ["label"]
    )[0]

    expression_id = int(
        np.argmax(output)
    )

    return EXPRESSIONS[
        expression_id
    ]


# --------------------------------------
# 13. Draw stylish maroon text
# --------------------------------------
def draw_stylish_text(
    pil_image,
    text,
    position,
    font
):

    draw = ImageDraw.Draw(
        pil_image
    )

    x, y = position

    # Calculate text size
    bbox = draw.textbbox(
        (x, y),
        text,
        font=font
    )

    left, top, right, bottom = bbox

    padding_x = 12
    padding_y = 7

    # ----------------------------------
    # Shadow
    # ----------------------------------
    shadow_offset = 4

    draw.rectangle(
        [
            left - padding_x + shadow_offset,
            top - padding_y + shadow_offset,
            right + padding_x + shadow_offset,
            bottom + padding_y + shadow_offset
        ],
        fill=(80, 80, 80)
    )


    # ----------------------------------
    # Background box
    # ----------------------------------
    draw.rectangle(
        [
            left - padding_x,
            top - padding_y,
            right + padding_x,
            bottom + padding_y
        ],
        fill=BOX_COLOR
    )


    # ----------------------------------
    # White outline
    # ----------------------------------
    draw.text(
        (x, y),
        text,
        font=font,
        fill=TEXT_COLOR,
        stroke_width=2,
        stroke_fill=OUTLINE_COLOR
    )

    return pil_image


# --------------------------------------
# 14. Start webcam
# --------------------------------------
camera = cv2.VideoCapture(
    0
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


if not camera.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    exit()


print()
print(
    "Facial Expression Recognition Started"
)
print(
    f"Hello {person_name}!"
)
print(
    "Press Q to quit"
)
print()


# --------------------------------------
# 15. Main real-time loop
# --------------------------------------
while True:

    success, frame = camera.read()

    if not success:

        print(
            "Could not read webcam."
        )

        break


    # ----------------------------------
    # Mirror webcam
    # ----------------------------------
    frame = cv2.flip(
        frame,
        1
    )


    # ----------------------------------
    # Tell YuNet current frame size
    # ----------------------------------
    height, width = frame.shape[:2]

    face_detector.setInputSize(
        (width, height)
    )


    # ----------------------------------
    # Detect faces
    # ----------------------------------
    _, faces = face_detector.detect(
        frame
    )


    # ----------------------------------
    # Process detected faces
    # ----------------------------------
    if faces is not None:

        for face in faces:

            # --------------------------
            # Face rectangle
            # --------------------------
            x = int(face[0])
            y = int(face[1])

            w = int(face[2])
            h = int(face[3])


            # --------------------------
            # Get five facial landmarks
            # --------------------------
            landmarks = face[
                4:14
            ].reshape(
                5,
                2
            )


            # --------------------------
            # Align face
            # --------------------------
            aligned_face = align_face(
                frame,
                landmarks
            )

            if aligned_face is None:
                continue


            # --------------------------
            # Predict expression
            # --------------------------
            expression = classify_expression(
                aligned_face
            )


            # --------------------------
            # Japanese expression
            # --------------------------
            japanese_expression = (
                JAPANESE_EXPRESSIONS[
                    expression
                ]
            )


            # --------------------------
            # Final sentence
            # Example:
            # Jaani, you are happy / 嬉しい
            # --------------------------
            display_text = (
                f"{person_name}, "
                f"you are "
                f"{expression.lower()} / "
                f"{japanese_expression}"
            )


            # --------------------------
            # Draw face rectangle
            # --------------------------
            cv2.rectangle(
                frame,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                FACE_BOX_COLOR,
                3
            )


            # --------------------------
            # Draw landmarks
            # --------------------------
            for point in landmarks:

                px = int(
                    point[0]
                )

                py = int(
                    point[1]
                )

                cv2.circle(
                    frame,
                    (px, py),
                    3,
                    LANDMARK_COLOR,
                    -1
                )


            # --------------------------
            # Convert OpenCV -> Pillow
            # --------------------------
            frame_pil = Image.fromarray(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
            )


            # --------------------------
            # Position text
            # --------------------------
            text_y = y - 45

            if text_y < 10:

                text_y = y + h + 10


            # --------------------------
            # Draw stylish text
            # --------------------------
            frame_pil = draw_stylish_text(
                frame_pil,
                display_text,
                (
                    max(x, 10),
                    text_y
                ),
                japanese_font
            )


            # --------------------------
            # Convert Pillow -> OpenCV
            # --------------------------
            frame = cv2.cvtColor(
                np.array(
                    frame_pil
                ),
                cv2.COLOR_RGB2BGR
            )


    # --------------------------------------
    # 16. Show result
    # --------------------------------------
    cv2.imshow(
        "Real-Time Facial Expression Recognition",
        frame
    )


    # --------------------------------------
    # 17. Press Q to quit
    # --------------------------------------
    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):

        break


# --------------------------------------
# 18. Cleanup
# --------------------------------------
camera.release()

cv2.destroyAllWindows()

print(
    "Program stopped."
)
