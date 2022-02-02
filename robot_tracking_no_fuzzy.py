import numpy as np
import cv2
import serial
import time
import math
from math import atan2, cos, sin, sqrt, pi, atan, degrees
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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

pointsList = [(0,0)] * 3
global_angle = 0
CM_TO_PIXEL = 32.0 / 640
sudut = 0

def gradient(pt1, pt2):
    if (pt2[0] - pt1[0]) == 0:
        return 0
    else:
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def get_angle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    sudut_radian = atan((m2 - m1) / (1 + (m2 * m1)))
    sudut_derajat = round(degrees(sudut_radian))

    # cv2.putText(img, str(sudut_derajat), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    return sudut_derajat

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
    return int(dist * CM_TO_PIXEL)

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    #print(p2)

    drawAxis(img, cntr, p1, (255, 255, 0), 9)
    drawAxis(img, cntr, p2, (0, 0, 255), 7)
    cv2.line(img, cntr, pointsList[1], (0, 0, 255), 3, cv2.LINE_AA)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]
    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv2.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Angle : " + str(-int(np.rad2deg(angle)) - 90) + " degree", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2 )

    points_array = []
    points_array.append(cntr)
    points_array.append(pointsList[1])
    points_array.append(p1)
    sudut_robot = get_angle(points_array)
    print(sudut_robot)
    cv2.putText(img, "Angle : " + str(sudut_robot) + " degree", (20, 240),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return sudut_robot

def fuzzy_robot(d, a):
    if d > 5:
        if -30 <= a < 30:
            VR = 150
            VL = 150
        elif -180 <= a < -30:
            VR = 30
            VL = 150
        elif 30 <= a <= 180:
            VR = 150
            VL = 30
        else:
            VR = 0
            VL = 0
    else:
        VR = 0
        VL = 0

    vel = []
    vel.append(VR)
    vel.append(VL)
    return vel

# def write_read(x):
#     arduino.write(bytes([int(x)]))
#     time.sleep(0.05)
#     data = arduino.readline()
#     return data

# ======================================================================================================================
# main section
# event handler
cv2.namedWindow("Optical Flow")  # must match the imshow 1st argument
cv2.setMouseCallback("Optical Flow", point_click)

cap = cv2.VideoCapture("videos/dataset-1.mp4")
# cap = cv2.VideoCapture(0)
# arduino = serial.Serial(port='COM10', baudrate=115200, timeout=.1)

while True:
    # start time to calculate FPS
    start = time.time()

    suc, frame = cap.read()

    cv2.circle(frame, point, radius, colour, lineWidth, cv2.FILLED)  # circle properties as arguments
    frame = cv2.resize(frame, (0, 0), fx=0.99, fy=0.99)  # this fx,fy value will be explained in post

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv2.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 3700 or 100000 < area:
            continue

        # Draw each contour only for visualisation purposes
        cv2.drawContours(frame, contours, i, (0, 0, 255), 2)
        sudut = getOrientation(c, frame)

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
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # Draw coordinate
        cv2.putText(img, "Coordinate : " + str(trajectories[0][0]), (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2 )
        cv2.putText(img, "Distance : " + str( distance(pointsList[1][0], pointsList[1][1], trajectories[0][0][0], trajectories[0][0][1]) ) + " cm", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Latest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 3, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])
        pointsList[0] = p
    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    d = distance(pointsList[1][0], pointsList[1][1], trajectories[0][0][0], trajectories[0][0][1])
    print(d)
    print(sudut)
    velo = fuzzy_robot(d, sudut)
    print(velo)
    # Kirim perintah ke Serial Bluetooth
    # VR = write_read(str(int(round(velo[0],0))))
    # VL = write_read(str(int(round(velo[1],0))))

    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "VR : " + str(round(velo[0], 2)) + " PWM", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "VL : " + str(round(velo[1], 2)) + " PWM", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Optical Flow', img)
    # cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()