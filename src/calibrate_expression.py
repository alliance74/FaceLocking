# src/calibrate_expression.py
"""
Live Expression & Eye State Calibration Tool.

Run:
    python -m src.calibrate_expression --cam 2

Keys:
    q : quit
"""
import argparse

import cv2

from .expression import ExpressionAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=2)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {args.cam} not opened.")

    analyzer = ExpressionAnalyzer(debug=False)

    print("Expression calibration. Smile / laugh / blink / frown.")
    print("Press 'q' to quit.\n")

    cv2.namedWindow("Expression Calibration", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = analyzer.analyze(frame)
        vis = frame.copy()

        if result:
            eye_state = "CLOSED" if result.eyes_closed else "OPEN"
            lines = [
                f"Expression: {result.expression}",
                f"Eyes: {eye_state}  EAR: {result.ear_avg:.3f} / thresh {result.ear_thresh:.3f}",
                f"Mouth lift: {result.mouth_norm_diff:+.3f}  width: {result.mouth_width_norm:.3f}  open: {result.mouth_open_norm:.3f}",
                f"Blinks: {result.blink_count}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(vis, line, (10, 32 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (10, 32 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Expression Calibration", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
