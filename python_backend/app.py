
import threading
import time

import cv2
from fastapi import FastAPI
from fastapi.responses import Response
from ultralytics import YOLO

app = FastAPI(title="Person Detector")

model = YOLO("yolo11n.pt")

lock = threading.Lock()

latest_frame = None
latest_count = 0
latest_people = []
camera_error = None
running = True


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

            # Run YOLO every 0.5 seconds, not on every frame.
            if now - last_detection >= 0.5:
                try:
                    result = model.predict(
                        frame,
                        classes=[0],
                        conf=0.4,
                        imgsz=640,
                        verbose=False,
                    )[0]

                    new_boxes = []
                    people = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = [
                                int(v)
                                for v in box.xyxy[0].tolist()
                            ]
                            confidence = float(box.conf[0])

                            new_boxes.append(
                                (x1, y1, x2, y2, confidence)
                            )

                            people.append({
                                "box": [x1, y1, x2, y2],
                                "confidence": round(confidence, 3),
                            })

                    cached_boxes = new_boxes

                    with lock:
                        latest_count = len(people)
                        latest_people = people
                        camera_error = None

                    last_detection = now

                except Exception as exc:
                    with lock:
                        camera_error = f"Detection error: {exc}"

            # Draw the latest detections on EVERY fresh camera frame.
            # This prevents boxes from appearing only twice per second.
            display_frame = frame.copy()

            for x1, y1, x2, y2, confidence in cached_boxes:
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                )

                label = f"INVALID {confidence:.2f}"

                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
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


@app.get("/")
def home():
    return {
        "message": "Person Detector is running",
        "camera": "Check /status",
    }


@app.get("/status")
def status():
    with lock:
        return {
            "count": latest_count,
            "people": latest_people,
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
