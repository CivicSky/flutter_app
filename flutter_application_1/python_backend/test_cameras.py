
import cv2
import time

for index in range(10):
    print(f"\nTesting camera {index}...")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera unavailable")
        continue

    success_count = 0

    for _ in range(10):
        ret, frame = cap.read()

        if ret and frame is not None:
            success_count += 1

        time.sleep(0.1)

    print(f"Successful frames: {success_count}/10")

    if success_count > 0:
        print(f"Camera {index} is WORKING")

    cap.release()

print("\nCamera testing complete.")
