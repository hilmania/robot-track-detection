import numpy as np
import cv2
import math
from math import atan2, cos, sin, sqrt, pi, atan, degrees

pointsList = [(0, 0)] * 3
# Create the circle
colour = (0, 0, 255)
lineWidth = -1  # -1 will result in filled circle
radius = 8
point = (0, 0)
CM_TO_PIXEL = 32.0 / 740

# function for detecting left mouse click
def point_click(event, x, y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pressed", x, y)
        point = (x, y)
        pointsList[1] = point


    sudut = get_angle(pointsList)
    print(sudut)

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

def gradient(pt1, pt2):
    if (pt2[0] - pt1[0]) == 0:
        return 0
    else:
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

def get_angle(pointsList):
    a = pointsList[-2]
    b = pointsList[-3]
    c = pointsList[-1]

    m1 = gradient(b, a)
    m2 = gradient(b, c)

    angle = atan((m2 - m1) / 1 + m2 * m1)
    angle = round(degrees(angle))
    if angle < 0:
        angle = 180 + angle
    return angle

# function for calculate distance
def distance(x1, y1, x2, y2):
    dist = math.sqrt(math.fabs(x2 - x1) ** 2 + math.fabs(y2 - y1) ** 2)
    return round(dist * CM_TO_PIXEL, 2)

cv2.namedWindow("Optical Flow")  # must match the imshow 1st argument
cv2.setMouseCallback("Optical Flow", point_click)
# Capturing video through webcam
cap = cv2.VideoCapture(0)

# Start a while loop
while (1):

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = cap.read()

    cv2.circle(imageFrame, point, radius, colour, lineWidth, cv2.FILLED)  # circle properties as arguments
    # imageFrame = cv2.resize(imageFrame, (0, 0), fx=0.99, fy=0.99)  # this fx,fy value will be explained in post

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([22, 93, 0])
    yellow_upper = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                                 mask=yellow_mask)

    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)

            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 255), 2)

            M = cv2.moments(contour)
            cx = int(M['m10'] // M['m00'])
            cy = int(M['m01'] // M['m00'])
            cv2.circle(imageFrame, (cx, cy), 3, (0, 255, 255), -1)

            drawAxis(imageFrame,(cx, cy), (cx, y), (255, 255, 0), 3)
            pointsList[0] = (cx, cy)
            pointsList[2] = (cx, y)
            cv2.line(imageFrame, (cx, cy), pointsList[1], (0, 0, 255), 3, cv2.LINE_AA)

    # Program Termination
    cv2.imshow("Optical Flow", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break