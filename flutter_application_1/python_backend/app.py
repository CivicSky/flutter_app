
import os
import threading
import time
import winsound
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from ultralytics import YOLO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

app = FastAPI(title="Smart Gate and Uniform Monitoring System")

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 640

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

# Load custom trained YOLO model
model = YOLO(str(MODEL_PATH))

print("Loaded model:", MODEL_PATH)
print("Model classes:", model.names)

# Shared state locks
frame_lock = threading.Lock()
raw_lock = threading.Lock()
detection_lock = threading.Lock()

# Video and detection state
latest_raw_frame = None
latest_frame = None
cached_boxes = []
latest_people = []
latest_count = 0
camera_error = None
running = True


# --------------------------------------------------
# DETECTION WORKER
# --------------------------------------------------

def detection_worker():
    global cached_boxes
    global latest_people
    global latest_count
    global camera_error

    print("YOLO detection worker started.")

    while running:

        with raw_lock:
            frame_to_detect = (
                None
                if latest_raw_frame is None
                else latest_raw_frame.copy()
            )

        if frame_to_detect is None:
            time.sleep(0.02)
            continue

        try:
            with torch.inference_mode():

                results = model.predict(
                    source=frame_to_detect,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    verbose=False,
                )

            result = results[0]

            new_boxes = []
            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    # Bounding box coordinates
                    x1, y1, x2, y2 = [
                        int(v)
                        for v in box.xyxy[0].tolist()
                    ]

                    # Class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = result.names[class_id]
                    class_name = str(class_name).strip().lower()

                    # ----------------------------------
                    # CLASSIFICATION
                    # ----------------------------------

                    if class_name == "valid":

                        is_valid = True
                        display_name = "Valid"
                        color = (0, 200, 0)

                        reason = (
                            "Prescribed clothing detected"
                        )

                    elif class_name == "invalid":

                        is_valid = False
                        display_name = "Invalid"
                        color = (0, 0, 255)

                        reason = (
                            "Prohibited or non-compliant clothing detected"
                        )

                    else:

                        is_valid = None
                        display_name = "Unknown"
                        color = (0, 165, 255)

                        reason = (
                            "Unrecognized clothing class"
                        )

                    # ----------------------------------
                    # SAVE BOUNDING BOX
                    # ----------------------------------

                    new_boxes.append((
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        display_name,
                        color
                    ))

                    # ----------------------------------
                    # SAVE DETECTION DATA
                    # ----------------------------------

                    detections.append({

                        "box": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],

                        "class_id": class_id,
                        "class_name": display_name,

                        "confidence": round(
                            confidence, 3
                        ),

                        "is_valid": is_valid,
                        "reason": reason,

                    })

            # Update shared detection state
            with detection_lock:

                cached_boxes = new_boxes
                latest_people = detections
                latest_count = len(detections)
                camera_error = None

        except Exception as exc:

            with detection_lock:
                camera_error = f"Detection error: {exc}"

            print("Detection error:", exc)

        time.sleep(0.03)


# --------------------------------------------------
# CAMERA WORKER
# --------------------------------------------------

def camera_worker():
    global latest_raw_frame
    global latest_frame
    global camera_error

    camera_index = int(os.getenv("CAMERA_INDEX", "0"))

    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW
    )

    # Try MSMF if DirectShow fails
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_MSMF
        )

    # Try the alternate camera index if both backends fail
    if not camera.isOpened():
        camera.release()

        fallback_index = 1 if camera_index == 0 else 0

        camera = cv2.VideoCapture(
            fallback_index,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(
                fallback_index,
                cv2.CAP_MSMF
            )

        if camera.isOpened():
            camera_index = fallback_index

    # Check camera availability
    if not camera.isOpened():
        camera.release()

        with detection_lock:
            camera_error = (
                "Cannot open webcam. Check camera permissions "
                "and ensure another app is not using it."
            )

        print(camera_error)
        return

    # Test whether the camera actually sends frames
    print("Camera successfully opened!")

    ret, test_frame = camera.read()

    if ret and test_frame is not None:
        print("Camera is sending frames!")
        print("Frame size:", test_frame.shape)
    else:
        print("ERROR: Camera opened but no frames received!")

    # Camera configuration
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    print(
        f"Webcam capture started on camera index {camera_index}."
    )

    try:

        while running:

            ok, frame = camera.read()

            if not ok:
                time.sleep(0.01)
                continue

            # Send latest frame to detection worker
            with raw_lock:
                latest_raw_frame = frame.copy()

            # Get latest detections
            with detection_lock:
                current_boxes = list(cached_boxes)

            # Draw detections
            display_frame = frame.copy()

            for (
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_name,
                color
            ) in current_boxes:

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"{class_name.upper()} {confidence:.2f}"
                )

                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # Encode frame as JPEG
            ok, encoded = cv2.imencode(
                ".jpg",
                display_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    75
                ],
            )

            if ok:

                with frame_lock:
                    latest_frame = encoded.tobytes()

            time.sleep(0.01)

    finally:

        camera.release()

        print("Webcam capture stopped.")


# --------------------------------------------------
# STARTUP AND SHUTDOWN
# --------------------------------------------------

@app.on_event("startup")
def start_camera():

    global running

    running = True

    # Start camera thread
    threading.Thread(
        target=camera_worker,
        daemon=True
    ).start()

    # Start YOLO detection thread
    threading.Thread(
        target=detection_worker,
        daemon=True
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

        "message": (
            "Smart Gate and Uniform Monitoring System is running"
        ),

        "model_classes": model.names,

        "confidence_threshold": CONFIDENCE_THRESHOLD,

        "image_size": IMAGE_SIZE,

        "endpoints": [
            "/status",
            "/frame",
            "/beep",
            "/docs"
        ],

    }


# --------------------------------------------------
# DETECTION STATUS
# --------------------------------------------------

@app.get("/status")
def status():

    with detection_lock:

        detections = list(latest_people)
        count = latest_count
        err = camera_error

    # Count valid and invalid clothing items
    valid_count = sum(
        1
        for d in detections
        if d["class_name"].lower() == "valid"
    )

    invalid_count = sum(
        1
        for d in detections
        if d["class_name"].lower() == "invalid"
    )

    has_invalid = invalid_count > 0
    has_valid = valid_count > 0

    # ----------------------------------------------
    # OVERALL COMPLIANCE STATUS
    # ----------------------------------------------

    if has_invalid:

        compliance_status = "Non-Compliant"

    elif has_valid:

        # A Valid clothing detection alone cannot
        # confirm that the complete outfit is compliant.
        compliance_status = "Needs Review"

    else:

        compliance_status = "No Detection"
        
    # Check camera frame availability
    with frame_lock:
        ready = latest_frame is not None

    return {

        "count": count,

        "people": detections,
        "detections": detections,

        "valid_count": valid_count,
        "invalid_count": invalid_count,

        "has_invalid": has_invalid,
        "has_valid": has_valid,

        "compliance_status": compliance_status,

        "camera_error": err,

        "ready": ready,

    }


# --------------------------------------------------
# BEEP ALERT
# --------------------------------------------------

@app.post("/beep")
def trigger_beep():

    try:

        threading.Thread(
            target=winsound.Beep,
            args=(1200, 250),
            daemon=True
        ).start()

        return {
            "status": "ok",
            "message": "Alert beep triggered"
        }

    except Exception as e:

        return {
            "status": "error",
            "error": str(e)
        }


# --------------------------------------------------
# LIVE CAMERA FRAME
# --------------------------------------------------

@app.get("/frame")
def frame():

    with frame_lock:
        image = latest_frame

    if image is None:

        return Response(
            content=b"Camera frame not available yet",
            status_code=503
        )

    return Response(
        content=image,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store"
        }
    )
