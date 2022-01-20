from __future__ import print_function
from basicmotiondetector import BasicMotionDetector
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2

print("[INFO] starting cameras...")
webcam = VideoStream(src=0).start()
# picam = VideoStream(usePiCamera=True).start()
picam = VideoStream(src=1).start()
time.sleep(2.0)

camMotion = BasicMotionDetector()
piMotion = BasicMotionDetector()
total = 0

while True:
    frames = []

    for (stream, motion) in zip((webcam, picam), (camMotion, piMotion)):
        frame = stream.read()
        frame = imutils.resize(frame, width=400)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        locs = motion.update(gray)

        if total < 32:
            frames.append(frame)
            continue

        if len(locs) > 0:
            (minX, minY) = (np.inf, np.inf)
            (maxX, maxY) = (-np.inf, -np.inf)

            for l in locs:
                (x, y, w, h) = cv2.boundingRect(l)
                (minX, maxX) = (min(minX, x), max(maxX, x + w))
                (minY, maxY) = (min(minY, y), max(maxY, y + h))

            cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 3)

        frames.append(frame)

        total += 1
        timestamp = datetime.datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")

        for (frame, name) in zip(frames, ("Webcam", "Picamera")):
            cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

print("[INFO] cleaning up...")
cv2.destroyAllWindows()
webcam.stop()
picam.stop()
