import cv2
import numpy as np


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
# 2. Model files
# --------------------------------------
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL = "facial_expression_recognition.onnx"


# --------------------------------------
# 3. Standard face landmark positions
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
# 4. Create face detector
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
# 5. Load expression model
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
# 6. Align face using 5 landmarks
# --------------------------------------
def align_face(frame, landmarks):

    landmarks = landmarks.astype(np.float32)

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
# 7. Classify facial expression
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

    return EXPRESSIONS[expression_id]


# --------------------------------------
# 8. Start webcam
# --------------------------------------
camera = cv2.VideoCapture(0)

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


print(
    "Facial Expression Recognition Started"
)

print(
    "Press Q to quit"
)


# --------------------------------------
# 9. Main real-time loop
# --------------------------------------
while True:

    success, frame = camera.read()

    if not success:

        print(
            "Could not read webcam."
        )

        break


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )


    # ----------------------------------
    # 10. Tell YuNet current frame size
    # ----------------------------------
    height, width = frame.shape[:2]

    face_detector.setInputSize(
        (width, height)
    )


    # ----------------------------------
    # 11. Detect faces
    # ----------------------------------
    _, faces = face_detector.detect(
        frame
    )


    # ----------------------------------
    # 12. Process detected faces
    # ----------------------------------
    if faces is not None:

        for face in faces:

            # Face rectangle
            x = int(face[0])
            y = int(face[1])

            w = int(face[2])
            h = int(face[3])


            # ----------------------------------
            # 13. Get five facial landmarks
            # ----------------------------------
            landmarks = face[
                4:14
            ].reshape(
                5,
                2
            )


            # ----------------------------------
            # 14. Align face
            # ----------------------------------
            aligned_face = align_face(
                frame,
                landmarks
            )

            if aligned_face is None:
                continue


            # ----------------------------------
            # 15. Predict expression
            # ----------------------------------
            expression = classify_expression(
                aligned_face
            )


            # ----------------------------------
            # 16. Draw face rectangle
            # ----------------------------------
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # ----------------------------------
            # 17. Draw expression text
            # ----------------------------------
            cv2.putText(
                frame,
                expression,
                (
                    x,
                    max(y - 10, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # ----------------------------------
            # 18. Draw landmarks
            # ----------------------------------
            for point in landmarks:

                px = int(point[0])
                py = int(point[1])

                cv2.circle(
                    frame,
                    (px, py),
                    3,
                    (255, 0, 0),
                    -1
                )


    # --------------------------------------
    # 19. Show result
    # --------------------------------------
    cv2.imshow(
        "Real-Time Facial Expression Recognition",
        frame
    )


    # --------------------------------------
    # 20. Press Q to quit
    # --------------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# --------------------------------------
# 21. Cleanup
# --------------------------------------
camera.release()

cv2.destroyAllWindows()

print(
    "Program stopped."
)