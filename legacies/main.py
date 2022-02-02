from imutils.video import FPS
import imutils
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi, atan, degrees
import numpy as np

pointsList = []

def drawAxis(frame, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
    ## [visualization1]


def getOrientation(pts, frame):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv.circle(frame, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(frame, cntr, p1, (255, 255, 0), 1)
    drawAxis(frame, cntr, p2, (0, 0, 255), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    #textbox = cv.rectangle(frame, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    #cv.putText(frame, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

    return angle

def mousePoints (event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv.line(frame,tuple(pointsList[round((size -1 ) / 3) * 3]), (x, y), (0, 0, 255), 2)
        cv.circle(frame, (x, y), 5, (0, 0, 255), cv.FILLED)
        pointsList.append([x, y])
        #print(pointsList)
        #print(x, y)
    return pointsList

def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

def getAngle(pointsList):
    #print("angle")
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(degrees(angR))
    print(angD)
    cv.putText(frame, str(angD) + " deg", (pt1[0] - 40, pt1[1] - 20), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    return angD
    

def calculateDistance (pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    # dist = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dist = sqrt((pt3[0] - pt1[0])**2 + (pt3[1] - pt1[1])**2)
    print(dist)
    cv.putText(frame, str(round(dist, 2)) + " px", (pt1[0] + 40, pt1[1] + 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return dist

# Define Tracker Optical Flow
tracker = cv.TrackerMedianFlow_create()

initBB = None

cap = cv.VideoCapture("../videos/dataset-1.mp4")

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
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        fps.update()
        fps.stop()

        info = [
            ("Tracker", "Median Flow Tracker"),
            ("Success", "Yes" if success else "NO"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, H - ((i * 20) + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.imshow("Frame", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("s"):
        initBB = cv.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
        fps = FPS().start()
    elif key == ord("p"):
        # cv.waitKey(0)
        img = frame

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Convert image to binary
        _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        # Find all the contours in the thresholded image
        contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        for i, c in enumerate(contours):

            # Calculate the area of each contour
            area = cv.contourArea(c)

            # Ignore contours that are too small or too large
            if area < 3700 or 100000 < area:
                continue

            # Draw each contour only for visualisation purposes
            cv.drawContours(img, contours, i, (0, 0, 255), 2)

            # Find the orientation of each shape
            getOrientation(c, img)

        while True:
            cv.imshow("Draw Point", img)
            cv.setMouseCallback('Draw Point', mousePoints)

            if len(pointsList) % 3 == 0 and len(pointsList) != 0:
                getAngle(pointsList)
                calculateDistance(pointsList)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord("e"):
                break
            
    elif key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()