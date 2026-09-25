# src/track_camera_check.py
"""
Phase 4 sanity check: confirm the turret-mounted camera (index 0) opens
and streams cleanly -- separate from your laptop webcam (index 1), which
the face-recognition pipeline already uses.

Run:
    python -m src.track_camera_check

Keys:
    q : quit
"""
import cv2


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera index 0 not opened. Check the USB connection, "
                            "or try index 2/3 if 0 is actually something else on this machine.")

    print("Tracking camera test (index 0). Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.imshow("Tracking Camera (index 0)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
