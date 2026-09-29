import cv2
import numpy as np


# -------------------------------------------------
# 1. COCO classes
# -------------------------------------------------
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# -------------------------------------------------
# 2. Settings
# -------------------------------------------------
MODEL_PATH = "yolov10n.onnx"

INPUT_SIZE = 640

# Lower threshold gives better detection
CONFIDENCE_THRESHOLD = 0.25


# -------------------------------------------------
# 3. Letterbox function
# -------------------------------------------------
def letterbox(image, size=640):

    height, width = image.shape[:2]

    # Keep aspect ratio
    scale = min(size / width, size / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    # Gray YOLO padding
    canvas = np.full(
        (size, size, 3),
        114,
        dtype=np.uint8
    )

    # Center image
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    return canvas, scale, pad_x, pad_y


# -------------------------------------------------
# 4. Load YOLO model
# -------------------------------------------------
print("Loading YOLO model...")

net = cv2.dnn.readNetFromONNX(
    MODEL_PATH
)

net.setPreferableBackend(
    cv2.dnn.DNN_BACKEND_OPENCV
)

net.setPreferableTarget(
    cv2.dnn.DNN_TARGET_CPU
)

print("Model loaded successfully.")


# -------------------------------------------------
# 5. Open webcam
# -------------------------------------------------
camera = cv2.VideoCapture(0)

# Request normal webcam resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


print("Real-Time Object Detection Started")
print("Press Q to stop")


# -------------------------------------------------
# 6. Main loop
# -------------------------------------------------
while True:

    success, frame = camera.read()

    if not success:

        print("Could not read webcam.")

        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # -------------------------------------------------
    # 7. Letterbox image
    # -------------------------------------------------
    input_image, scale, pad_x, pad_y = letterbox(
        frame,
        INPUT_SIZE
    )


    # -------------------------------------------------
    # 8. Convert to neural network input
    # -------------------------------------------------
    blob = cv2.dnn.blobFromImage(
        input_image,
        scalefactor=1 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False
    )


    # -------------------------------------------------
    # 9. Run YOLO
    # -------------------------------------------------
    net.setInput(blob)

    output = net.forward()


    # YOLOv10 output:
    # [1, 300, 6]

    detections = np.squeeze(output)


    # -------------------------------------------------
    # 10. Process detections
    # -------------------------------------------------
    for detection in detections:

        x1, y1, x2, y2, confidence, class_id = detection[:6]


        if confidence < CONFIDENCE_THRESHOLD:
            continue


        class_id = int(class_id)

        if class_id < 0 or class_id >= len(CLASSES):
            continue


        # -------------------------------------------------
        # 11. Remove letterbox padding
        # -------------------------------------------------
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale

        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale


        # Convert to integer
        x1 = int(x1)
        y1 = int(y1)

        x2 = int(x2)
        y2 = int(y2)


        # -------------------------------------------------
        # 12. Keep coordinates inside image
        # -------------------------------------------------
        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))

        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))


        # Ignore invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue


        # -------------------------------------------------
        # 13. Object name
        # -------------------------------------------------
        object_name = CLASSES[class_id]

        label = (
            f"{object_name} "
            f"{confidence * 100:.0f}%"
        )


        # -------------------------------------------------
        # 14. Draw bounding box
        # -------------------------------------------------
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # -------------------------------------------------
        # 15. Draw label
        # -------------------------------------------------
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # -------------------------------------------------
    # 16. Display result
    # -------------------------------------------------
    cv2.imshow(
        "Real-Time Object Detection",
        frame
    )


    # -------------------------------------------------
    # 17. Q = quit
    # -------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------------------------------
# 18. Cleanup
# -------------------------------------------------
camera.release()

cv2.destroyAllWindows()

print("Program stopped.")