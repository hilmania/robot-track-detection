#from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np

# Define Tracker Optical Flow
tracker = cv2.TrackerMedianFlow_create()

initBB = None

cap = cv2.VideoCapture("videos/dataset-1.mp4")

fps = None

while True:
    ret, frame = cap.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=1400)
    H, W = frame.shape[:2]
    if initBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        fps.update()
        fps.stop()

        info = [
            ("Tracker", "Median Flow Tracker"),
            ("Success", "Yes" if success else "NO"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
        fps = FPS().start()

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


