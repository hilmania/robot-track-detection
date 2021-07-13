import cv2
import sys
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

# Define Tracker Optical Flow
trackerFlow = cv2.TrackerMedianFlow_create()

cap = cv2.VideoCapture("videos/dataset-1.mp4")

# Exit if video not opened.
if not cap.isOpened():
    print("Could not open video")
    sys.exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print('Cannot read video file')
    sys.exit()

bbox = (287, 23, 86, 320)

bbox = cv2.selectROI(frame, False)

ret = trackerFlow.init(frame, bbox)

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #  === Euclidean Distance Tracker ===
    # height, width, _ = frame.shape
    #
    # # Extract Region of interest
    # roi = frame[:,:]
    #
    # # 1. Object Detection
    # mask = object_detector.apply(roi)
    # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # detections = []
    # for cnt in contours:
    #     # Calculate area and remove small elements
    #     area = cv2.contourArea(cnt)
    #     if area > 100:
    #         #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         detections.append([x, y, w, h])
    #
    # # 2. Object Tracking
    # boxes_ids = tracker.update(detections)
    # for box_id in boxes_ids:
    #     x, y, w, h, id = box_id
    #     # cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    #     cv2.putText(roi, str(x), (x - 15, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    #     cv2.putText(roi, str(y), (x + 60, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #
    # cv2.imshow("roi", roi)
    # cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)

    # === End of Euclidean Tracker ===

    # === Median Flow Tracker ===
    # Start timer
    timer = cv2.getTickCount()

    ret, bbox = trackerFlow.update(frame)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ret:
        # Tracking success
        p1 = (float(bbox[0]), float(bbox[1]))
        p2 = (float(bbox[0], bbox[2]), float(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display Tracker
        cv2.putText(frame, "Median Flow Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : ", + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)
    # === End of Median Flow Tracker ===

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()