
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import Response
from ultralytics import YOLO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

app = FastAPI(title="Smart Gate and Uniform Monitoring System")

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

# Load YOUR trained uniform model, not yolo11n.pt
model = YOLO(str(MODEL_PATH))

print("Loaded model:", MODEL_PATH)
print("Model classes:", model.names)

lock = threading.Lock()

latest_frame = None
latest_count = 0
latest_people = []
camera_error = None
running = True


# --------------------------------------------------
# CAMERA AND UNIFORM DETECTION
# --------------------------------------------------

def camera_worker():
    global latest_frame, latest_count, latest_people
    global camera_error

    CAMERA_INDEX = 0
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

    if not camera.isOpened():
        camera.release()

        with lock:
            camera_error = (
                "Cannot open webcam. Check camera permissions "
                "and ensure another app is not using it."
            )

        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    last_detection = 0.0
    cached_boxes = []

    print(f"Webcam {CAMERA_INDEX} started.")

    try:
        while running:
            ok, frame = camera.read()

            if not ok:
                time.sleep(0.02)
                continue

            now = time.time()

            # Run uniform detection every 0.5 seconds.
            if now - last_detection >= 0.5:
                try:
                    result = model.predict(
                        source=frame,
                        conf=0.25,
                        imgsz=640,
                        verbose=False,
                    )[0]

                    new_boxes = []
                    detections = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = [
                                int(v)
                                for v in box.xyxy[0].tolist()
                            ]

                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            class_name = result.names[class_id]

                            # Green for Valid; red for Invalid.
                            if class_name.lower() == "valid":
                                color = (0, 200, 0)
                            else:
                                color = (0, 0, 255)

                            new_boxes.append(
                                (
                                    x1, y1, x2, y2,
                                    confidence, class_name, color
                                )
                            )

                            detections.append({
                                "box": [x1, y1, x2, y2],
                                "class_id": class_id,
                                "class_name": class_name,
                                "confidence": round(confidence, 3),
                            })

                    cached_boxes = new_boxes

                    with lock:
                        latest_count = len(detections)
                        latest_people = detections
                        camera_error = None

                    last_detection = now

                except Exception as exc:
                    with lock:
                        camera_error = f"Detection error: {exc}"

                    print("Detection error:", exc)
                    time.sleep(0.1)

            # Draw detections on every camera frame.
            display_frame = frame.copy()

            for (
                x1, y1, x2, y2,
                confidence, class_name, color
            ) in cached_boxes:

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                label = f"{class_name.upper()} {confidence:.2f}"

                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            ok, encoded = cv2.imencode(
                ".jpg",
                display_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            if ok:
                with lock:
                    latest_frame = encoded.tobytes()

    finally:
        camera.release()
        print("Webcam stopped.")


# --------------------------------------------------
# STARTUP AND SHUTDOWN
# --------------------------------------------------

@app.on_event("startup")
def start_camera():
    global running

    running = True

    threading.Thread(
        target=camera_worker,
        daemon=True,
    ).start()


@app.on_event("shutdown")
def stop_camera():
    global running

    running = False


# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Smart Gate and Uniform Monitoring System is running",
        "model_classes": model.names,
        "endpoints": ["/status", "/frame"],
    }


@app.get("/status")
def status():
    with lock:
        return {
            "count": latest_count,
            "people": latest_people,
            "detections": latest_people,
            "camera_error": camera_error,
            "ready": latest_frame is not None,
        }


@app.get("/frame")
def frame():
    with lock:
        image = latest_frame

    if image is None:
        return Response(
            content=b"Camera frame not available yet",
            status_code=503,
        )

    return Response(
        content=image,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
