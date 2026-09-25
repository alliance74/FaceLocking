# src/camera.py
"""Webcam smoke test. Tries several camera indices if needed.

Run:
    python -m src.camera
    python -m src.camera --cam 2

Keys:
    q : quit
"""
import argparse

import cv2


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    # warm-up + brightness check (black frames = wrong device)
    mean = 0.0
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            mean = float(frame.mean())
    if mean < 5.0:
        cap.release()
        return None
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=None, help="Camera index (0/1/2...)")
    args = parser.parse_args()

    cap = None
    used = None

    if args.cam is not None:
        cap = open_camera(args.cam)
        used = args.cam
        if cap is None:
            # still show it even if dark, so user can see something
            cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError(f"Camera {args.cam} failed to open.")
    else:
        for i in range(4):
            print(f"Trying camera index {i}...")
            cap = open_camera(i)
            if cap is not None:
                used = i
                break
        if cap is None:
            raise RuntimeError(
                "No working camera found (indices 0-3). "
                "Unplug/replug the USB cam, close Zoom/Teams, then retry:\n"
                "  .venv\\Scripts\\python.exe -m src.camera --cam 2"
            )

    print(f"Camera test using index {used}. Press 'q' to quit.")
    print("Tip: drag the window corner to enlarge, or click maximize — OpenCV is not exclusive fullscreen.")
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    sized = False
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        if not sized:
            h, w = frame.shape[:2]
            cv2.resizeWindow("Camera Test", w, h)
            sized = True

        cv2.putText(
            frame,
            f"cam index={used}  (press q to quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Camera Test", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
