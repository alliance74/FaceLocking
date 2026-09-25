# src/face_center_track.py
"""
Phase 5 sanity check: detect a face on the turret camera (index 0) and
compute its horizontal pixel offset from the frame center -- this is the
raw tracking error signal the servo control loop will use next.

No servo movement yet. Just visualizing the offset to confirm it behaves
correctly (positive when your face is right of center, negative when left)
before trusting it to drive a motor.

Run:
    python -m src.face_center_track

Keys:
    q : quit
"""
import cv2

from .haar_5pt import Haar5ptDetector


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Tracking camera (index 0) not opened.")

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)

    print("Face center tracking test (index 0). Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        frame_cx = W // 2

        vis = frame.copy()

        # vertical line marking true frame center
        cv2.line(vis, (frame_cx, 0), (frame_cx, H), (255, 0, 0), 1)

        faces = det.detect(frame, max_faces=1)

        if faces:
            f = faces[0]
            face_cx = (f.x1 + f.x2) // 2
            face_cy = (f.y1 + f.y2) // 2

            offset_x = face_cx - frame_cx  # negative = face is LEFT of center, positive = RIGHT

            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            cv2.circle(vis, (face_cx, face_cy), 5, (0, 0, 255), -1)
            cv2.line(vis, (frame_cx, face_cy), (face_cx, face_cy), (0, 255, 255), 2)

            direction = "LEFT" if offset_x < 0 else ("RIGHT" if offset_x > 0 else "CENTER")
            cv2.putText(vis, f"offset_x: {offset_x:+d}px ({direction})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Center Tracking (index 0)", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
