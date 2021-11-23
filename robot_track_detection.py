import numpy as np
import cv2
import time
import math
from math import atan2, cos, sin, sqrt, pi, atan, degrees

# Create the circle
colour = (0, 0, 255)
# colour = (0, 255, 81)
lineWidth = -1  # -1 will result in filled circle
radius = 15
point = (0, 0)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

pointsList = [None] * 3

# function for detecting left mouse click
def point_click(event, x, y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pressed", x, y)
        point = (x, y)
        pointsList[1] = point

# function for calculate distance
def distance(x1, y1, x2, y2):
    """
    Calculate distance between two points
    """
    dist = math.sqrt(math.fabs(x2 - x1) ** 2 + math.fabs(y2 - y1) ** 2)
    return dist

# function for finding color match
def find_color1(frame):
    """
    Filter "frame" for HSV bounds for color1 (inplace, modifies frame) & return coordinates of the object with that color
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_lowerbound = np.array([139, 149, 131])  # replace THIS LINE w/ your hsv lowerb
    hsv_upperbound = np.array([179, 255, 219])  # replace THIS LINE w/ your hsv upperb
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    res = cv2.bitwise_and(frame, frame, mask=mask)  # filter inplace
    cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxcontour = max(cnts, key=cv2.contourArea)

        # Find center of the contour
        M = cv2.moments(maxcontour)
        if M['m00'] > 0 and cv2.contourArea(maxcontour) > 1000:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy), True
        else:
            return (700, 700), False  # faraway point
    else:
        return (700, 700), False  # faraway point

# function for finding color match
def find_color2(frame):
    """
    Filter "frame" for HSV bounds for color1 (inplace, modifies frame) & return coordinates of the object with that color
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_lowerbound = np.array([101, 152, 92])  # replace THIS LINE w/ your hsv lowerb
    hsv_upperbound = np.array([149, 255, 243])  # replace THIS LINE w/ your hsv upperb
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxcontour = max(cnts, key=cv2.contourArea)

        # Find center of the contour
        M = cv2.moments(maxcontour)
        if M['m00'] > 0 and cv2.contourArea(maxcontour) > 2000:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy), True  # True
        else:
            return (700, 700), True  # faraway point
    else:
        return (700, 700), True  # faraway point


def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def get_angle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(degrees(angR))
    print(angD)
    cv2.putText(img, str(angD), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    return angD

# ======================================================================================================================
# main section
# event handler
cv2.namedWindow("Optical Flow")  # must match the imshow 1st argument
cv2.setMouseCallback("Optical Flow", point_click)

cap = cv2.VideoCapture("videos/dataset-1.mp4")
# cap = cv2.VideoCapture(0)

while True:
    # start time to calculate FPS
    start = time.time()

    suc, frame = cap.read()

    cv2.circle(frame, point, radius, colour, lineWidth, cv2.FILLED)  # circle properties as arguments
    frame = cv2.resize(frame, (0, 0), fx=0.99, fy=0.99)  # this fx,fy value will be explained in post

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            # cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), 3)
        trajectories = new_trajectories

        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Latest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])
        # print(p)
        pointsList[0] = p
    frame_idx += 1
    prev_gray = frame_gray

    print(pointsList)
    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()